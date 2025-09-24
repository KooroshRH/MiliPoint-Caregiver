import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

class FocalLoss(nn.Module):
    """
    Focal Loss for dealing with class imbalance
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MixupAugmentation:
    """
    Mixup augmentation for point cloud data
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, x, y):
        batch_size = x.size(0)
        
        # Generate mixup coefficients
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = torch.tensor(lam, device=x.device).view(-1, 1, 1)
        
        # Create indices for shuffling the batch
        indices = torch.randperm(batch_size, device=x.device)
        
        # Perform mixup
        mixed_x = lam * x + (1 - lam) * x[indices]
        
        if y.dim() == 1:  # For classification
            return mixed_x, y, y[indices], lam.squeeze()
        else:  # For keypoint regression
            mixed_y = lam.view(-1, 1, 1) * y + (1 - lam).view(-1, 1, 1) * y[indices]
            return mixed_x, mixed_y

class AdvancedModelWrapper(pl.LightningModule):
    """
    Advanced model wrapper with improved training techniques
    """
    def __init__(
            self,
            model,
            learning_rate=5e-4,
            weight_decay=1e-4,
            epochs=300,
            optimizer='adamw',
            scheduler='cosine_warmup',
            loss_type='default',
            mixup_alpha=0.2,
            use_mixup=False,
            label_smoothing=0.1):
        super().__init__()
        self.model = model
        self.num_classes = self.model.num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_type = loss_type
        self.use_mixup = use_mixup
        self.mixup = MixupAugmentation(mixup_alpha) if use_mixup else None
        self.label_smoothing = label_smoothing
        
        # Define appropriate loss function
        if self.num_classes is None:  # keypoint regression
            self.loss = nn.MSELoss()
            self.metric_name = 'mle'
            self.metric = mean_localization_error
        else:  # classification
            if loss_type == 'focal':
                self.loss = FocalLoss(gamma=2.0)
            elif loss_type == 'default':
                self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.metric_name = 'acc'
            self.metric = acc
            
        # Training and validation tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # Prediction tracking for evaluation
        self.ys = []
        self.y_hats = []
        
        # Auto log gradients
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        opt = self.optimizers()
        
        # Manually zero gradients
        opt.zero_grad()
        
        # Apply mixup if enabled for classification
        if self.use_mixup and self.num_classes is not None:
            x, y_a, y_b, lam = self.mixup(x, y)
            y_hat = self.forward(x)
            loss = lam * self.loss(y_hat, y_a) + (1-lam) * self.loss(y_hat, y_b)
            metric = lam * self.metric(y_hat, y_a) + (1-lam) * self.metric(y_hat, y_b)
        # Apply mixup if enabled for keypoints
        elif self.use_mixup and self.num_classes is None:
            x, y = self.mixup(x, y)
            y_hat = self.forward(x)
            loss = self.loss(y_hat, y)
            metric = self.metric(y_hat, y)
        else:
            # Standard forward pass
            y_hat = self.forward(x)
            loss = self.loss(y_hat, y)
            metric = self.metric(y_hat, y)
            
        # Manual backpropagation
        self.manual_backward(loss)
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Optimizer step
        opt.step()
        
        # Update schedulers
        sch = self.lr_schedulers()
        if self.scheduler == 'one_cycle':
            sch.step()
        
        # Log metrics
        self.log_dict(
            {"loss": loss, f'train_{self.metric_name}': metric},
            on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            
        # Log learning rate
        self.log('lr', opt.param_groups[0]['lr'], on_step=False, on_epoch=True)
        
        return {"loss": loss, "metric": metric}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        
        # Calculate loss and metrics
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)
        
        # Log validation metrics
        self.log_dict(
            {"val_loss": loss, f'val_{self.metric_name}': metric},
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
        # Track validation loss for best model selection
        self.val_losses.append(loss)
        
        return {"val_loss": loss, "val_metric": metric}
    
    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        
        # Calculate loss and metrics
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)
        
        # Save predictions for analysis
        self.ys.append(y)
        self.y_hats.append(y_hat)
        
        # Enhanced logging for classification tasks
        if self.metric_name == 'acc':
            # Calculate top-3 accuracy
            top3 = torch.topk(y_hat, min(3, y_hat.size(1)), dim=1)[1]
            top3_acc = (top3 == y.unsqueeze(-1)).float().sum()/x.shape[0]
            
            self.log_dict(
                {
                    "test_loss": loss,
                    f'test_{self.metric_name}': metric,
                    f'test_top3_{self.metric_name}': top3_acc
                },
                on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            self.log_dict(
                {
                    "test_loss": loss,
                    f'test_{self.metric_name}': metric
                },
                on_step=False, on_epoch=True, prog_bar=False, logger=True)
                
        return {"test_loss": loss, "test_metric": metric}
    
    def predict_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        return self(x), y
    
    def configure_optimizers(self):
        # Select optimizer
        if self.optimizer == 'adamw':
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'adam':
            opt = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'sgd':
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
            
        # Select scheduler
        if self.scheduler == 'cosine_warmup':
            scheduler = CosineAnnealingWarmRestarts(
                opt,
                T_0=10,
                T_mult=2,
                eta_min=self.learning_rate * 0.01
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss"
            }
        elif self.scheduler == 'one_cycle':
            scheduler = OneCycleLR(
                opt,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos'
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")
            
        return {"optimizer": opt, "lr_scheduler": scheduler_config}
    
    def on_validation_epoch_end(self):
        if len(self.val_losses):
            val_loss = torch.stack(self.val_losses).mean()
            
            # Track best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                
            # Reset validation loss tracking
            self.val_losses = []
            
    def on_test_epoch_end(self):
        # Concatenate predictions from all batches
        self.ys = torch.cat(self.ys)
        self.y_hats = torch.cat(self.y_hats)
        
        # For classification tasks, perform detailed analysis
        if self.num_classes is not None:
            y_pred = torch.argmax(self.y_hats, dim=1)
            
            # Calculate F1 score and other metrics
            f1 = f1_score(self.ys.cpu(), y_pred.cpu(), average='macro')
            precision, recall, f1_per_class, support = precision_recall_fscore_support(
                self.ys.cpu(), y_pred.cpu(), average=None
            )
            
            # Calculate confusion matrix
            cm = confusion_matrix(self.ys.cpu(), y_pred.cpu())
            
            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(f'confusion_matrix_epoch_{self.current_epoch}.png')
            plt.close()
            
            # Apply class merging if it's an action recognition task
            if hasattr(self, 'merge_classes') and self.merge_classes:
                merge_map = {
                    0: 1, 1: 3, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 
                    8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 3, 15: 1, 
                    16: 3, 17: 1, 18: 1, 19: 3, 20: 1, 21: 2, 22: 1, 23: 3, 
                    24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 0
                }
                
                merged_ys = torch.tensor([merge_map[int(y)] for y in self.ys])
                merged_y_preds = torch.tensor([merge_map[int(y)] for y in y_pred])
                
                # Calculate metrics for merged classes
                f1_merged = f1_score(merged_ys.cpu(), merged_y_preds.cpu(), average='macro')
                precision_merged, recall_merged, f1_per_class_merged, support_merged = precision_recall_fscore_support(
                    merged_ys.cpu(), merged_y_preds.cpu(), average=None
                )
                
                # Calculate confusion matrix for merged classes
                cm_merged = confusion_matrix(merged_ys.cpu(), merged_y_preds.cpu())
                
                # Plot confusion matrix for merged classes
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm_merged, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix (Merged Classes)')
                plt.savefig(f'confusion_matrix_merged_epoch_{self.current_epoch}.png')
                plt.close()
                
                # Log merged metrics
                print(f"\nTest F1 score (Merged Classes): {f1_merged}")
                print(f"Merged classes confusion matrix:\n{cm_merged}")
            
            # Log general metrics
            print(f"\nTest F1 score: {f1}")
            print(f"Confusion matrix:\n{cm}")
            
            # Log per-class metrics
            for i in range(len(precision)):
                print(f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_per_class[i]:.4f}, Support={support[i]}")
                
        # Reset prediction tracking
        self.ys = []
        self.y_hats = []

def mean_localization_error(x, y):
    """Calculate the mean localization error for keypoint prediction"""
    dist = (x-y).pow(2).sum(-1).sqrt().mean()
    return dist

def acc(x, y):
    """Calculate the accuracy for classification tasks"""
    acc = (torch.argmax(x, axis=1) == y).float().sum()/x.shape[0]
    return acc

def advanced_train(
        model,
        train_loader, val_loader,
        optimizer='adamw', 
        learning_rate=1e-3, 
        weight_decay=1e-4,
        epochs=300,
        scheduler='cosine_warmup',
        loss_type='default',
        use_mixup=False,
        mixup_alpha=0.2,
        label_smoothing=0.1,
        early_stopping_patience=15,
        save_path='./checkpoints',
        experiment_name='experiment',
        gpus=1,
        precision=16,
        merge_classes=False):
    """
    Advanced training function with various optimization strategies
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer type ('adamw', 'adam', 'sgd')
        learning_rate: Initial learning rate
        weight_decay: Weight decay coefficient
        epochs: Maximum number of training epochs
        scheduler: Learning rate scheduler ('cosine_warmup', 'one_cycle')
        loss_type: Loss function type ('default', 'focal')
        use_mixup: Whether to use mixup augmentation
        mixup_alpha: Alpha parameter for mixup
        label_smoothing: Label smoothing factor for classification
        early_stopping_patience: Number of epochs to wait before early stopping
        save_path: Path to save model checkpoints
        experiment_name: Name for the experiment
        gpus: Number of GPUs to use
        precision: Training precision (16 or 32)
        merge_classes: Whether to apply class merging for action recognition
    
    Returns:
        Best validation loss or metric value
    """
    # Create model wrapper
    wrapper = AdvancedModelWrapper(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_type=loss_type,
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        label_smoothing=label_smoothing
    )
    
    # Set merge_classes flag if needed
    if merge_classes:
        wrapper.merge_classes = True
    
    # Define metric for model checkpointing
    metric = f'val_{wrapper.metric_name}'
    
    # Determine mode for checkpointing
    if 'mle' in metric:
        mode = 'min'
    elif 'acc' in metric:
        mode = 'max'
    else:
        raise ValueError(f'Unknown metric {metric}')
    
    # Create callbacks
    callbacks = [
        # Save best model based on validation metric
        ModelCheckpoint(
            save_top_k=1,
            monitor=metric,
            mode=mode,
            filename="best_{epoch}",
            dirpath=save_path,
            save_last=True,
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor=metric,
            patience=early_stopping_patience,
            mode=mode,
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Setup logger
    logger = TensorBoardLogger(save_dir="tb_logs", name=experiment_name)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else None,
        precision=precision,
        log_every_n_steps=10,
        deterministic=False,  # Set to True for reproducibility, but slower
    )
    
    # Train the model
    trainer.fit(wrapper, train_loader, val_loader)
    
    # Return best validation metric
    if mode == 'min':
        return wrapper.best_val_loss
    else:
        return wrapper.best_val_acc
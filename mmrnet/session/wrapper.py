import pytorch_lightning as pl
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class ModelWrapper(pl.LightningModule):

    def __init__(
            self,
            model,
            learning_rate=5e-4,
            weight_decay=1e-5,
            epochs=200,
            optimizer=None):
        super().__init__()
        self.model = model
        self.num_classes = self.model.num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if self.num_classes is None:      # keypoints
            self.loss = torch.nn.MSELoss()
            self.metric_name = 'mle'
            self.metric = mean_localization_error
        else:                           # iden or action
            self.loss = torch.nn.CrossEntropyLoss()
            self.metric_name = 'acc'
            self.metric = acc
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

        self.best_val_loss = 10e9

        self.ys = []
        self.y_hats = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)

        # loss
        self.log_dict(
            {"loss": loss, f'{self.metric_name}': metric},
            on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)
        # val_loss
        self.log_dict(
            {"val_loss": loss, f'val_{self.metric_name}': metric},
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_losses.append(loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        metric = self.metric(y_hat, y)

        self.ys.append(y)
        self.y_hats.append(y_hat)

        # Log metrics
        if self.metric_name == 'acc':
            # compute top-3
            top3 = torch.topk(y_hat, 3, dim=1)[1]
            top3_acc = (top3 == y.unsqueeze(-1)).float().sum()/x.shape[0]
            
            self.log_dict(
                {
                    "test_loss": loss, 
                    f'test_{self.metric_name}': metric,
                    f'test_top3_{self.metric_name}': top3_acc},
                on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            self.log_dict(
                {
                    "test_loss": loss, 
                    f'test_{self.metric_name}': metric},
                on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {"test_loss": loss}

    def predict_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        return self(x), y

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        elif self.optimizer in ['sgd_warmup', 'sgd']:
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True)
            if self.optimizer == 'sgd':
                scheduler = CosineAnnealingLR(
                    opt, T_max=self.epochs, eta_min=0.0)
        return {
            "optimizer": opt,
            "lr_scheduler":  scheduler}
    
    def on_validation_epoch_end(self):
        if len(self.val_losses):
            val_loss = torch.stack(self.val_losses).mean()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            self.val_losses = []

    def on_test_epoch_end(self):
        self.ys = torch.cat(self.ys)
        self.y_hats = torch.cat(self.y_hats)

        # Calculate F1 score before merging
        f1_before_merge = f1_score(self.ys.cpu(), torch.argmax(self.y_hats, axis=1).cpu(), average='macro')
        
        # Define the mapping for merging classes into 4 groups
        merge_map = {
            0: 1, 1: 3, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 
            8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 3, 15: 1, 
            16: 3, 17: 1, 18: 1, 19: 3, 20: 1, 21: 2, 22: 1, 23: 3, 
            24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 0
        }
        merged_ys = torch.tensor([merge_map[int(y)] for y in self.ys])
        merged_y_hats = torch.tensor([merge_map[int(y)] for y in torch.argmax(self.y_hats, axis=1)])

        # Calculate F1 score for merged classes
        f1_merged = f1_score(merged_ys.cpu(), merged_y_hats.cpu(), average='macro')
        
        # Calculate accuracy for merged classes
        accuracy_merged = (merged_ys == merged_y_hats).float().mean().item()
        
        # Compute confusion matrix for merged classes
        cm_merged = confusion_matrix(merged_ys.cpu(), merged_y_hats.cpu())

        print(f"\nConfusion Matrix (Merged Classes): \n{cm_merged}")
        
        # Plot confusion matrix for merged classes
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_merged, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Merged Classes)')
        plt.savefig(f'confusion_matrix_merged_epoch_{self.current_epoch}.png')
        plt.close()

        print(f"\n\nTest F1 score (Before Merging): {f1_before_merge}")
        print(f"\nTest F1 score (Merged Classes): {f1_merged}")
        print(f"\nTest Accuracy (Merged Classes): {accuracy_merged}")

def mean_localization_error(x, y):
    dist = (x-y).pow(2).sum(-1).sqrt().mean()
    return dist

def acc(x, y):
    acc = (torch.argmax(x, axis=1) == y).float().sum()/x.shape[0]
    return acc

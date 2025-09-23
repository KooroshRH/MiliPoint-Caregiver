import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool, global_mean_pool
from .dgcnn import DGCNN
from .attdgcnn import AttDGCNN
from .hybrid_pointnet import HybridPointNet

class EnsembleModel(nn.Module):
    """
    Ensemble model that combines multiple point cloud architectures
    for improved prediction accuracy through model averaging
    """
    def __init__(self, model_types=None, weights=None, k=30, aggr='max', info=None):
        super(EnsembleModel, self).__init__()
        self.num_classes = info['num_classes']
        self.info = info
        
        # Default to using the three main models if not specified
        if model_types is None:
            model_types = ['dgcnn', 'attdgcnn', 'hybrid']
            
        # Initialize model weights if not provided (equal weighting)
        if weights is None:
            weights = [1.0/len(model_types)] * len(model_types)
        else:
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)
        
        # Create sub-models
        self.models = nn.ModuleList()
        for model_type in model_types:
            if model_type.lower() == 'dgcnn':
                self.models.append(DGCNN(k=k, aggr=aggr, info=info))
            elif model_type.lower() == 'attdgcnn':
                self.models.append(AttDGCNN(k=k, aggr=aggr, info=info))
            elif model_type.lower() == 'hybrid':
                self.models.append(HybridPointNet(k=k, aggr=aggr, info=info))
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        # Model fusion approach
        self.fusion = nn.Sequential(
            nn.Linear(len(model_types) * (512 if self.num_classes is None else 128), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output layers based on task
        if self.num_classes is None:  # keypoint prediction
            num_points = info['num_keypoints']
            self.num_points = num_points
            point_branches = {}
            for i in range(num_points):
                point_branches[f'branch_{i}'] = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 3)
                )
            self.output = nn.ModuleDict(point_branches)
        else:  # classification (identification or action)
            self.output = nn.Linear(128, self.num_classes)
            
    def get_model_features(self, x):
        """Extract intermediate features from each model"""
        features = []
        
        for model in self.models:
            with torch.no_grad():
                # For DGCNN model
                if isinstance(model, DGCNN):
                    batchsize = x.shape[0]
                    npoints = x.shape[1]
                    x_reshaped = x.reshape((batchsize * npoints, 3))
                    batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
                    
                    xs = []
                    for conv in model.conv:
                        x_reshaped = conv(x_reshaped, batch)
                        xs.append(x_reshaped)
                    
                    x4 = model.lin1(torch.cat(xs, dim=1))
                    feature = global_max_pool(x4, batch)
                
                # For AttDGCNN model
                elif isinstance(model, AttDGCNN):
                    batchsize = x.shape[0]
                    npoints = x.shape[1]
                    x_reshaped = x.reshape((batchsize * npoints, 3))
                    batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
                    
                    model_features = []
                    for conv in model.edge_convs:
                        x_reshaped = conv(x_reshaped, batch)
                        model_features.append(x_reshaped)
                        
                    x_concat = torch.cat(model_features, dim=1)
                    x_concat = model.temporal_module(x_concat, batch)
                    
                    x_max = global_max_pool(x_concat, batch)
                    x_mean = global_mean_pool(x_concat, batch)
                    feature = x_max + x_mean
                
                # For HybridPointNet model
                elif isinstance(model, HybridPointNet):
                    batchsize = x.shape[0]
                    npoints = x.shape[1]
                    
                    # Reshape input
                    x_reshaped = x.reshape(batchsize, npoints, 3)
                    x_reshaped = x_reshaped.transpose(2, 1)  # [B, 3, N]
                    
                    # Apply input transformation
                    trans = model.stn(x_reshaped)
                    x_reshaped = x_reshaped.transpose(2, 1)  # [B, N, 3]
                    x_reshaped = torch.bmm(x_reshaped, trans)
                    x_reshaped = x_reshaped.transpose(2, 1)  # [B, 3, N]
                    
                    # Extract local features (PointNet style)
                    local_feat = model.local_features(x_reshaped)
                    
                    # Apply feature transformation
                    trans_feat = model.fstn(local_feat)
                    local_feat = local_feat.transpose(2, 1)  # [B, N, 64]
                    local_feat = torch.bmm(local_feat, trans_feat)
                    local_feat = local_feat.transpose(2, 1)  # [B, 64, N]
                    
                    feature = model.feature_aggregator(local_feat).view(batchsize, -1)
                
                features.append(feature)
                
        return torch.cat(features, dim=1)
        
    def forward(self, x):
        """Forward pass with late fusion of model outputs"""
        # Base model predictions
        base_preds = []
        base_features = []
        
        for i, model in enumerate(self.models):
            # Get model prediction
            pred = model(x)
            base_preds.append(pred)
            
            # Store intermediate feature (from second-to-last layer)
            if self.num_classes is None:  # keypoint prediction
                # Extract features from the last common layer
                feature = model.point_features(x)  # This would need to be customized based on model architecture
                base_features.append(feature)
            else:  # classification
                # For classification, use the features before the final layer
                # This would need custom hooks for each model type
                # For simplicity, we use the model outputs here
                feature = pred
                base_features.append(feature)
        
        # Weighted fusion of base model predictions
        if self.training:
            # During training, just use the weighted average for efficiency
            result = None
            for i, pred in enumerate(base_preds):
                if result is None:
                    result = self.weights[i] * pred
                else:
                    result += self.weights[i] * pred
            return result
        else:
            # During inference, also consider feature fusion for better results
            # Concatenate features from all models
            fused_features = torch.cat(base_features, dim=1)
            
            # Apply fusion network
            fused_features = self.fusion(fused_features)
            
            # Task-specific output
            if self.num_classes is None:  # keypoint prediction
                # For keypoint prediction, generate keypoints from fused features
                keypoints = []
                for i in range(self.num_points):
                    keypoints.append(self.output[f'branch_{i}'](fused_features))
                fused_output = torch.stack(keypoints, dim=1)
                
                # Average with base model predictions
                result = None
                for i, pred in enumerate(base_preds):
                    if result is None:
                        result = self.weights[i] * pred
                    else:
                        result += self.weights[i] * pred
                        
                # Combine base models and fusion model (giving more weight to fusion)
                final_output = 0.7 * fused_output + 0.3 * result
            else:  # classification
                # For classification, generate class probabilities from fused features
                fused_output = self.output(fused_features)
                
                # Average with base model predictions
                result = None
                for i, pred in enumerate(base_preds):
                    if result is None:
                        result = self.weights[i] * pred
                    else:
                        result += self.weights[i] * pred
                        
                # Combine base models and fusion model
                final_output = torch.softmax(0.7 * fused_output + 0.3 * result, dim=1)
                
            return final_output

# Utility function to load model from checkpoint
def load_ensemble_from_checkpoint(checkpoint_paths, model_types, weights=None, k=30, aggr='max', info=None):
    """
    Load an ensemble model from saved checkpoints of individual models
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        model_types: List of model types corresponding to checkpoints
        weights: Optional weights for ensemble averaging
        k: k-parameter for graph construction
        aggr: Aggregation method for graph features
        info: Model configuration information
        
    Returns:
        Loaded ensemble model with weights from checkpoints
    """
    assert len(checkpoint_paths) == len(model_types), "Number of checkpoint paths must match number of model types"
    
    # Create ensemble model
    ensemble = EnsembleModel(model_types=model_types, weights=weights, k=k, aggr=aggr, info=info)
    
    # Load weights for each model
    for i, path in enumerate(checkpoint_paths):
        checkpoint = torch.load(path, map_location='cpu')
        model_state_dict = checkpoint['state_dict']
        
        # Remove 'model.' prefix if present (from pytorch lightning)
        clean_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith('model.'):
                clean_state_dict[key[6:]] = value
            else:
                clean_state_dict[key] = value
                
        # Load weights into corresponding model
        ensemble.models[i].load_state_dict(clean_state_dict)
        
    return ensemble
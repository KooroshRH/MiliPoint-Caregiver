import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool, global_mean_pool
from torch_scatter import scatter_max

class STNkd(nn.Module):
    """Spatial Transformer Network for k-dimensional inputs"""
    def __init__(self, k=3):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k*self.k).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class LocalFeatureAggregator(nn.Module):
    """Aggregates local features using different pooling strategies"""
    def __init__(self, in_channels, out_channels):
        super(LocalFeatureAggregator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.conv(x)
        
        # Get multiple types of pooled features
        x_max = torch.max(x, dim=2, keepdim=True)[0]
        x_mean = torch.mean(x, dim=2, keepdim=True)
        x_std = torch.std(x, dim=2, keepdim=True)
        
        # Concatenate different pooling results
        x_global = torch.cat([x_max, x_mean, x_std], dim=1)
        return x_global

class DGCNNModule(nn.Module):
    """DGCNN module for dynamic graph feature extraction"""
    def __init__(self, in_channels, out_channels, k=20, aggr='max'):
        super(DGCNNModule, self).__init__()
        self.edge_conv = DynamicEdgeConv(
            MLP([in_channels * 2, out_channels, out_channels]),
            k=k, aggr=aggr
        )
        
    def forward(self, x, batch):
        return self.edge_conv(x, batch)

class HybridPointNet(nn.Module):
    """
    Hybrid model combining aspects of PointNet, DGCNN, and transformer mechanisms
    for robust point cloud processing with enhanced feature extraction
    """
    def __init__(self, k=30, aggr='max', info=None):
        super().__init__()
        self.num_classes = info['num_classes']
        self.info = info
        self.k = k
        
        # Input transformation with STN
        self.stn = STNkd(k=3)
        
        # Local features - PointNet style
        self.local_features = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Feature transformation for higher dimensions
        self.fstn = STNkd(k=64)
        
        # DGCNN modules for hierarchical feature extraction
        self.dgcnn1 = DGCNNModule(64, 64, k=k, aggr=aggr)
        self.dgcnn2 = DGCNNModule(64, 128, k=k, aggr=aggr)
        self.dgcnn3 = DGCNNModule(128, 256, k=k, aggr=aggr)
        
        # Additional pointwise MLP for feature enhancement
        self.point_features = nn.Sequential(
            nn.Conv1d(64+64+128+256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Local feature aggregation
        self.feature_aggregator = LocalFeatureAggregator(256, 128)
        
        # Global feature processor
        self.global_features = nn.Sequential(
            nn.Linear(128*3, 512),  # 3 types of pooling
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
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
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 3)
                )
            self.output = nn.ModuleDict(point_branches)
        else:  # classification (identification or action)
            self.output = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_classes)
            )
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, data):
        batchsize = data.shape[0]
        npoints = data.shape[1]
        
        # Reshape input
        x = data.reshape(batchsize, npoints, 3)
        x = x.transpose(2, 1)  # [B, 3, N]
        
        # Apply input transformation
        trans = self.stn(x)
        x = x.transpose(2, 1)  # [B, N, 3]
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)  # [B, 3, N]
        
        # Extract local features (PointNet style)
        local_feat = self.local_features(x)
        
        # Apply feature transformation
        trans_feat = self.fstn(local_feat)
        local_feat = local_feat.transpose(2, 1)  # [B, N, 64]
        local_feat = torch.bmm(local_feat, trans_feat)
        local_feat = local_feat.transpose(2, 1)  # [B, 64, N]
        
        # Prepare for DGCNN modules
        x = local_feat.transpose(2, 1).reshape(-1, local_feat.size(1))  # [B*N, 64]
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
        
        # Apply DGCNN modules
        dgcnn_feat1 = self.dgcnn1(x, batch)
        dgcnn_feat2 = self.dgcnn2(dgcnn_feat1, batch)
        dgcnn_feat3 = self.dgcnn3(dgcnn_feat2, batch)
        
        # Reshape and concatenate multi-level features
        all_feats = [
            x.view(batchsize, npoints, -1).transpose(2, 1),  # original features
            dgcnn_feat1.view(batchsize, npoints, -1).transpose(2, 1),  # level 1
            dgcnn_feat2.view(batchsize, npoints, -1).transpose(2, 1),  # level 2
            dgcnn_feat3.view(batchsize, npoints, -1).transpose(2, 1)   # level 3
        ]
        x = torch.cat(all_feats, dim=1)  # [B, 64+64+128+256, N]
        
        # Apply point features MLP
        x = self.point_features(x)  # [B, 256, N]
        
        # Aggregate local features
        x_global = self.feature_aggregator(x)  # [B, 128*3, 1]
        x_global = x_global.view(batchsize, -1)  # [B, 128*3]
        
        # Process global features
        x_global = self.global_features(x_global)  # [B, 256]
        
        # Task-specific output
        if self.num_classes is None:  # keypoint prediction
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](x_global))
            y = torch.stack(y, dim=1)
        else:  # classification
            y = self.output(x_global)
            
        return y
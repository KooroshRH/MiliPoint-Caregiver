import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool, global_mean_pool
from torch_scatter import scatter_max, scatter_mean

class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for channel attention"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, points)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y

class AttEdgeConv(nn.Module):
    """Enhanced Edge Convolution with attention mechanisms"""
    def __init__(self, in_channels, out_channels, k=20, aggr='max'):
        super(AttEdgeConv, self).__init__()
        self.k = k
        self.aggr = aggr
        
        # Feature transformation MLP
        self.mlp = MLP([in_channels * 2, out_channels, out_channels, out_channels])
        
        # Channel attention
        self.se = SELayer(out_channels)
        
        # Optional spatial attention
        self.spatial_att = SpatialAttention()
        
        # Residual connection when dimensions match
        self.use_residual = in_channels == out_channels
        if not self.use_residual:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, batch):
        # Apply edge convolution
        edge_conv = DynamicEdgeConv(self.mlp, self.k, self.aggr)
        x_out = edge_conv(x, batch)
        
        # Shape for attention: [batch_size, channels, points]
        batch_size = batch.max().item() + 1
        x_reshaped = x_out.view(batch_size, -1, x_out.size(1)).transpose(1, 2)
        
        # Apply attention mechanisms
        x_att = self.se(x_reshaped)
        x_att = self.spatial_att(x_att)
        
        # Reshape back
        x_att = x_att.transpose(1, 2).reshape(-1, x_out.size(1))
        
        # Residual connection
        if self.use_residual:
            x_res = x
        else:
            x_res = self.res_conv(x.view(batch_size, -1, x.size(1)).transpose(1, 2))
            x_res = x_res.transpose(1, 2).reshape(-1, x_out.size(1))
            
        return x_att + x_res

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module for global feature refinement"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TemporalModule(nn.Module):
    """Module for handling temporal information in point clouds"""
    def __init__(self, dim, stacks=None):
        super(TemporalModule, self).__init__()
        self.stacks = stacks
        if stacks:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            )
    
    def forward(self, x, batch):
        if not self.stacks:
            return x
            
        # Reshape to handle temporal dimension
        batch_size = batch.max().item() + 1
        x = x.view(batch_size, -1, x.size(1)).transpose(1, 2)  # [B, C, N]
        
        # Apply temporal convolution
        x = self.temporal_conv(x)
        
        # Reshape back
        x = x.transpose(1, 2).reshape(-1, x.size(1))
        return x

class AttDGCNN(nn.Module):
    """
    Attention-enhanced Dynamic Graph CNN
    Incorporates attention mechanisms, residual connections, and temporal modeling
    """
    def __init__(self, k=30, aggr='max', info=None):
        super().__init__()
        self.num_classes = info['num_classes']
        self.info = info
        
        # Network parameters
        self.k = k
        self.aggr = aggr
        
        # Basic input features
        input_dim = 3  # x, y, z coordinates
        
        # Enhanced edge convolution blocks with attention
        self.edge_convs = nn.ModuleList([
            AttEdgeConv(input_dim, 64, k=k, aggr=aggr),
            AttEdgeConv(64, 64, k=k, aggr=aggr),
            AttEdgeConv(64, 128, k=k, aggr=aggr),
            AttEdgeConv(128, 256, k=k, aggr=aggr)
        ])
        
        # Temporal module if stacks are used
        self.temporal_module = TemporalModule(512, info.get('stacks', None))
        
        # Global pooling: concatenate mean and max for better feature aggregation
        self.pool_dim = 512  # 64 + 64 + 128 + 256
        
        # Multi-head self-attention for global feature refinement
        self.self_attention = MultiHeadSelfAttention(self.pool_dim)
        
        # Feature transformation layers
        self.point_features = nn.Sequential(
            nn.Linear(self.pool_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
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
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3)
                )
            self.output = nn.ModuleDict(point_branches)
        else:  # classification (identification or action)
            self.output = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_classes)
            )
            
        # Initialize weights
        self.apply(self._init_weights)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, data):
        batchsize = data.shape[0]
        npoints = data.shape[1]
        x = data.reshape((batchsize * npoints, 3))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
        
        # Apply edge convolutions with attention
        features = []
        for conv in self.edge_convs:
            x = conv(x, batch)
            features.append(x)
            
        # Concatenate features from all layers
        x = torch.cat(features, dim=1)
        
        # Apply temporal modeling if available
        x = self.temporal_module(x, batch)
        
        # Global pooling (max + mean pooling for better feature representation)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_global = x_max + x_mean
        
        # Apply self-attention for global feature refinement
        x_global = x_global.unsqueeze(1)  # [B, 1, C]
        x_global = self.self_attention(x_global).squeeze(1)  # [B, C]
        
        # Transform features
        x_global = self.point_features(x_global)
        
        # Task-specific output
        if self.num_classes is None:  # keypoint prediction
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](x_global))
            y = torch.stack(y, dim=1)
        else:  # classification
            y = self.output(x_global)
            
        return y
"""
DeepGCN with Auxiliary Features Support
Modified from deepgcn.py to support full auxiliary data (XYZ + auxiliary channels)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MLP, knn_graph, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter


class DeepGCNLayer(nn.Module):
    """
    A single DeepGCN layer with residual connection and auxiliary feature support
    """
    def __init__(self, in_channels, out_channels, k=16, aggr='max', norm='batch', act='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.aggr = aggr

        # Edge function (MLP) - input is concatenation of source and target features
        self.edge_mlp = MLP([in_channels * 2, out_channels, out_channels],
                           norm=norm, act=act, plain_last=False)

        # Normalization and activation
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Identity()

        # Residual connection projection if dimensions don't match
        if in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = None

    def forward(self, x, pos, batch):
        """
        Args:
            x: (N, in_channels) node features (includes auxiliary data)
            pos: (N, 3) node positions
            batch: (N,) batch indices
        Returns:
            (N, out_channels) updated node features
        """
        identity = x

        # Build k-NN graph based on positions
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)

        # Gather features (includes auxiliary data)
        x_i = x[edge_index[0]]  # Source nodes
        x_j = x[edge_index[1]]  # Target nodes

        # Edge features: concatenate source and target
        edge_feat = torch.cat([x_i, x_j], dim=-1)

        # Apply edge MLP
        edge_feat = self.edge_mlp(edge_feat)

        # Aggregate to nodes
        if self.aggr == 'max':
            x_out = scatter(edge_feat, edge_index[0], dim=0, dim_size=x.size(0), reduce='max')
        elif self.aggr == 'mean':
            x_out = scatter(edge_feat, edge_index[0], dim=0, dim_size=x.size(0), reduce='mean')
        elif self.aggr == 'add':
            x_out = scatter(edge_feat, edge_index[0], dim=0, dim_size=x.size(0), reduce='add')
        else:
            x_out = scatter(edge_feat, edge_index[0], dim=0, dim_size=x.size(0), reduce='max')

        # Normalization
        x_out = self.norm(x_out)

        # Residual connection (res+)
        if self.res_proj is not None:
            identity = self.res_proj(identity)

        x_out = x_out + identity

        # Activation
        x_out = self.act(x_out)

        return x_out


class DeepGCN_Aux(nn.Module):
    """
    DeepGCN model with auxiliary feature support

    Input format: (B, N, C) where C = 7 (XYZ + 4 auxiliary channels)
    Output format: (B, num_classes) for classification

    Memory-optimized version with:
    - Reduced number of layers (7 instead of 14)
    - Smaller k-NN neighborhood (9 instead of 16)
    - Gradient checkpointing support
    """
    def __init__(self, info=None, in_channels=15, k=9, num_layers=7, channels=64, aggr='max'):
        super().__init__()
        self.num_classes = info['num_classes']
        self.in_channels = in_channels
        self.k = k
        self.num_layers = num_layers

        # Stem: project input (including auxiliary features) to initial feature dimension
        self.stem = MLP([in_channels, channels, channels], norm='batch', act='relu')

        # Deep GCN layers (reduced from 14 to 7 for memory efficiency)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Progressive channel expansion
            in_ch = channels
            out_ch = channels
            # Expand channels at layer 2 instead of layer 3.5
            if i == num_layers // 3:
                out_ch = channels * 2

            self.layers.append(
                DeepGCNLayer(in_ch, out_ch, k=k, aggr=aggr, norm='batch', act='relu')
            )
            channels = out_ch

        # Global pooling
        self.global_pool_type = 'max'  # or 'mean'

        # Classification head
        if self.num_classes is None:
            # Keypoint estimation
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([channels, 256, 128, 3],
                                                    dropout=0.5, norm=None)
            self.output = nn.ModuleDict(point_branches)
        else:
            # Classification
            self.output = MLP([channels, 256, 128, self.num_classes],
                            dropout=0.5, norm=None)

    def forward(self, data):
        point_cloud, frame_signals = data
        # point_cloud   : (B, T, N, 6)
        # frame_signals : (B, T, 9)
        B, T, N, _ = point_cloud.shape

        fs = frame_signals.unsqueeze(2).expand(-1, -1, N, -1)   # (B, T, N, 9)
        x = torch.cat([point_cloud, fs], dim=-1)                 # (B, T, N, 15)
        x = x.reshape(B * T * N, self.in_channels)
        batch = torch.arange(B * T, device=x.device).repeat_interleave(N)

        pos = x[:, :3]
        x = self.stem(x)

        for layer in self.layers:
            x = layer(x, pos, batch)

        if self.global_pool_type == 'max':
            x = global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)            # (B*T, channels)

        x = x.view(B, T, -1).mean(dim=1)              # (B, channels)

        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.output(x)

        return y

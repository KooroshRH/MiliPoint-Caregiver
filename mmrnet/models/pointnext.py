"""
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
Based on: https://github.com/guochengqian/PointNeXt

PointNeXt is an improved version of PointNet++ with:
- InvResMLP blocks (Inverted Residual MLP)
- Improved sampling and grouping strategies
- Better normalization and activation functions
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MLP, fps, global_max_pool, radius, knn


class InvResMLP(nn.Module):
    """Inverted Residual MLP block used in PointNeXt"""
    def __init__(self, in_channels, expansion=4, use_res=True):
        super().__init__()
        self.use_res = use_res
        mid_channels = in_channels * expansion

        self.conv1 = nn.Linear(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Linear(mid_channels, in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.act = nn.GELU()  # PointNeXt uses GELU

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_res:
            out = out + identity
        return self.act(out)


class PointNextSetAbstraction(nn.Module):
    """Set Abstraction module for PointNeXt with improved local aggregation"""
    def __init__(self, npoint, radius_val, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius_val
        self.nsample = nsample
        self.group_all = group_all

        # MLP for processing grouped points
        last_channel = in_channel + 3  # +3 for relative position
        mlp_layers = []
        for out_channel in mlp:
            mlp_layers.append(nn.Linear(last_channel, out_channel))
            mlp_layers.append(nn.BatchNorm1d(out_channel))
            mlp_layers.append(nn.GELU())
            last_channel = out_channel

        self.mlp_convs = nn.ModuleList(mlp_layers[::3])
        self.mlp_bns = nn.ModuleList(mlp_layers[1::3])
        self.act = nn.GELU()

        # Inverted Residual MLP for post-processing
        self.post_mlp = InvResMLP(mlp[-1])

    def forward(self, xyz, features, batch):
        """
        Args:
            xyz: (N, 3) point positions
            features: (N, C) point features
            batch: (N,) batch indices
        Returns:
            new_xyz: (N', 3) sampled point positions
            new_features: (N', C') aggregated features
            new_batch: (N',) new batch indices
        """
        if self.group_all:
            # Global pooling
            if features is not None:
                features = torch.cat([xyz, features], dim=1)
            else:
                features = xyz

            # Global max pooling
            new_features = global_max_pool(features, batch)
            new_xyz = xyz.new_zeros((new_features.size(0), 3))
            new_batch = torch.arange(new_features.size(0), device=batch.device)

        else:
            # Sample points using FPS
            idx = fps(xyz, batch, ratio=self.npoint)
            new_xyz = xyz[idx]
            new_batch = batch[idx]

            # Query neighbors using radius search
            row, col = radius(xyz, new_xyz, self.radius, batch, new_batch,
                            max_num_neighbors=self.nsample)
            edge_index = torch.stack([col, row], dim=0)

            # Relative positions
            grouped_xyz = xyz[edge_index[1]] - new_xyz[edge_index[0]]

            # Group features and concatenate with relative positions
            if features is not None:
                grouped_features = features[edge_index[1]]
                grouped_points = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                # Only use relative positions when there are no features
                grouped_points = grouped_xyz

            # Apply MLPs
            for conv, bn in zip(self.mlp_convs, self.mlp_bns):
                grouped_points = self.act(bn(conv(grouped_points)))

            # Max pooling within each group
            new_features = global_max_pool(grouped_points, edge_index[0])

            # Post-processing with InvResMLP
            new_features = self.post_mlp(new_features)

        return new_xyz, new_features, new_batch


class PointNext(nn.Module):
    """
    PointNeXt model for point cloud classification

    Input format: (B, N, 3) for vanilla version (XYZ only)
    Output format: (B, num_classes) for classification
    """
    def __init__(self, info=None):
        super().__init__()
        self.num_classes = info['num_classes']

        # PointNeXt architecture
        # SA1: 512 points, radius 0.2, 32 samples
        self.sa1 = PointNextSetAbstraction(
            npoint=0.5, radius_val=0.2, nsample=32,
            in_channel=0, mlp=[32, 32, 64]
        )

        # SA2: 128 points, radius 0.4, 64 samples
        self.sa2 = PointNextSetAbstraction(
            npoint=0.25, radius_val=0.4, nsample=64,
            in_channel=64, mlp=[64, 64, 128]
        )

        # SA3: Global pooling
        self.sa3 = PointNextSetAbstraction(
            npoint=None, radius_val=None, nsample=None,
            in_channel=128, mlp=[128, 256, 512], group_all=True
        )

        # Classification head
        if self.num_classes is None:
            # Keypoint estimation
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([512, 256, 128, 3], dropout=0.5, norm=None)
            self.output = nn.ModuleDict(point_branches)
        else:
            # Classification
            self.output = MLP([512, 256, 128, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        """
        Args:
            data: (B, N, C) where C >= 3
        Returns:
            (B, num_classes) or (B, num_keypoints, 3)
        """
        batchsize = data.shape[0]
        npoints = data.shape[1]

        # Extract XYZ coordinates only (first 3 channels)
        data = data[:, :, :3]

        # Reshape to (B*N, 3)
        xyz = data.reshape((batchsize * npoints, 3))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(xyz.device)

        # PointNeXt forward pass
        xyz1, features1, batch1 = self.sa1(xyz, None, batch)
        xyz2, features2, batch2 = self.sa2(xyz1, features1, batch1)
        xyz3, features3, batch3 = self.sa3(xyz2, features2, batch2)

        # Output head
        if self.num_classes is None:
            # Keypoint estimation
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](features3))
            y = torch.stack(y, dim=1)
        else:
            # Classification
            y = self.output(features3)

        return y

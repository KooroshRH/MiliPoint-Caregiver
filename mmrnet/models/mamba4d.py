"""
Mamba4D: Efficient 4D Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models
Based on: https://arxiv.org/abs/2405.14338 (CVPR 2025)

Key innovations:
- Disentangled spatial-temporal processing
- Intra-frame Spatial Mamba for local spatial patterns
- Inter-frame Temporal Mamba for long-range temporal dependencies
- Point tubes via spatio-temporal KNN
- Linear complexity for long sequences

Adapted for sparse radar point clouds (T=40 frames, N=22 points)
Requires: mamba-ssm, causal-conv1d
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MLP, knn, global_max_pool

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Mamba4D will not be available.")


class PointEmbedding(nn.Module):
    """
    Initial point embedding using mini-PointNet per frame.
    """
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, embed_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: (B*T, N, C) point features
        Returns:
            (B*T, N, embed_dim) embedded features
        """
        x = x.transpose(1, 2)  # (B*T, C, N)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)  # (B*T, N, embed_dim)
        return x


class SpatialMambaBlock(nn.Module):
    """
    Intra-frame Spatial Mamba block.
    Processes points within each frame using Mamba SSM.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B*T, N, D) per-frame point features
        Returns:
            (B*T, N, D) transformed features
        """
        return x + self.dropout(self.mamba(self.norm(x)))


class TemporalMambaBlock(nn.Module):
    """
    Inter-frame Temporal Mamba block.
    Processes temporal sequences across frames.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) temporal sequence
        Returns:
            (B, T, D) transformed sequence
        """
        return x + self.dropout(self.mamba(self.norm(x)))


class CrossFramePooling(nn.Module):
    """
    Cross-frame temporal pooling to aggregate short-term local structures.
    Uses a small temporal window to pool features across nearby frames.
    """
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, N, D) spatio-temporal features
        Returns:
            (B, T, N, D) temporally smoothed features
        """
        B, T, N, D = x.shape
        # Reshape for temporal pooling: (B*N, D, T)
        x = x.permute(0, 2, 3, 1).reshape(B * N, D, T)
        x = self.pool(x)
        x = x.reshape(B, N, D, T).permute(0, 3, 1, 2)  # (B, T, N, D)
        return self.norm(x)


class IntraFrameSpatialMamba(nn.Module):
    """
    Intra-frame Spatial Mamba module.
    Processes spatial patterns within each frame independently.
    """
    def __init__(self, dim, num_blocks=4, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpatialMambaBlock(dim, d_state, d_conv, expand, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Args:
            x: (B*T, N, D) per-frame features
        Returns:
            (B*T, N, D) spatially processed features
        """
        for block in self.blocks:
            x = block(x)
        return x


class InterFrameTemporalMamba(nn.Module):
    """
    Inter-frame Temporal Mamba module.
    Captures long-range temporal dependencies across all frames.
    """
    def __init__(self, dim, num_blocks=4, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalMambaBlock(dim, d_state, d_conv, expand, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, T, D) frame-level features
        Returns:
            (B, T, D) temporally processed features
        """
        for block in self.blocks:
            x = block(x)
        return x


def get_spatial_order(xyz, order_type='xyz'):
    """
    Get point ordering based on coordinate sorting.
    Multiple orderings capture different spatial structures.
    """
    B_T, N, _ = xyz.shape
    device = xyz.device

    if order_type == 'xyz':
        # Sort by x, then y, then z
        keys = xyz[:, :, 0] * 1e6 + xyz[:, :, 1] * 1e3 + xyz[:, :, 2]
    elif order_type == 'zyx':
        keys = xyz[:, :, 2] * 1e6 + xyz[:, :, 1] * 1e3 + xyz[:, :, 0]
    elif order_type == 'yxz':
        keys = xyz[:, :, 1] * 1e6 + xyz[:, :, 0] * 1e3 + xyz[:, :, 2]
    else:
        # Default: no reordering
        return torch.arange(N, device=device).unsqueeze(0).expand(B_T, -1)

    return torch.argsort(keys, dim=1)


class Mamba4D(nn.Module):
    """
    Mamba4D for 4D point cloud video understanding.

    Input format: (B, T, N, 3) for vanilla version (XYZ only)
                  or (B, N, 3) which gets reshaped assuming temporal stacking
    Output format: (B, num_classes) for classification

    Architecture:
    - Point embedding via mini-PointNet
    - Intra-frame Spatial Mamba (processes within each frame)
    - Cross-frame temporal pooling (short-term aggregation)
    - Inter-frame Temporal Mamba (long-range temporal modeling)
    - Global pooling + classification head
    """
    def __init__(
        self,
        info=None,
        in_channels=3,
        embed_dim=128,
        spatial_blocks=4,
        temporal_blocks=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        num_frames=40,  # Expected number of frames
        points_per_frame=22  # Expected points per frame
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required for Mamba4D. Install with: pip install mamba-ssm")

        self.num_classes = info['num_classes']
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.points_per_frame = points_per_frame

        # Point embedding
        self.point_embed = PointEmbedding(in_channels, embed_dim)

        # Learnable spatial positional encoding
        self.spatial_pos = nn.Parameter(torch.zeros(1, points_per_frame, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        # Learnable temporal positional encoding
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        # Intra-frame Spatial Mamba
        self.spatial_mamba = IntraFrameSpatialMamba(
            embed_dim, num_blocks=spatial_blocks,
            d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
        )

        # Cross-frame temporal pooling
        self.cross_frame_pool = CrossFramePooling(embed_dim, kernel_size=3)

        # Inter-frame Temporal Mamba
        self.temporal_mamba = InterFrameTemporalMamba(
            embed_dim, num_blocks=temporal_blocks,
            d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout
        )

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        if self.num_classes is None:
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([embed_dim, 128, 64, 3], dropout=0.5, norm=None)
            self.output = nn.ModuleDict(point_branches)
        else:
            self.output = MLP([embed_dim, 256, 128, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        """
        Args:
            data: (B, T, N, C) 4D point cloud video
                  or (B, T*N, C) flattened temporal data
        Returns:
            (B, num_classes) classification logits
        """
        # Handle different input formats
        if data.dim() == 3:
            # Assume flattened: (B, T*N, C) -> (B, T, N, C)
            B, TN, C = data.shape
            T = self.num_frames
            N = self.points_per_frame
            if TN == T * N:
                data = data.view(B, T, N, C)
            else:
                # Try to infer
                N = TN // T if TN % T == 0 else self.points_per_frame
                T = TN // N
                data = data.view(B, T, N, C)
        else:
            B, T, N, C = data.shape

        device = data.device

        # Extract XYZ coordinates
        xyz = data[:, :, :, :3]  # (B, T, N, 3)

        # Reshape for per-frame processing: (B*T, N, 3)
        xyz_flat = xyz.reshape(B * T, N, 3)

        # Point embedding
        x = self.point_embed(xyz_flat)  # (B*T, N, embed_dim)

        # Add spatial positional encoding
        if N <= self.spatial_pos.shape[1]:
            x = x + self.spatial_pos[:, :N, :]
        else:
            # Interpolate if needed
            x = x + self.spatial_pos[:, :N % self.spatial_pos.shape[1], :].repeat(1, N // self.spatial_pos.shape[1] + 1, 1)[:, :N, :]

        # Apply spatial ordering (multiple orderings for robustness)
        # Use xyz ordering
        order = get_spatial_order(xyz_flat, 'xyz')
        x_ordered = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Intra-frame Spatial Mamba
        x_spatial = self.spatial_mamba(x_ordered)  # (B*T, N, embed_dim)

        # Reorder back
        inverse_order = torch.argsort(order, dim=1)
        x_spatial = torch.gather(x_spatial, 1, inverse_order.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Reshape to (B, T, N, D) for temporal processing
        x_spatial = x_spatial.view(B, T, N, self.embed_dim)

        # Cross-frame temporal pooling
        x_pooled = self.cross_frame_pool(x_spatial)  # (B, T, N, embed_dim)

        # Per-frame global pooling (max pool over points)
        x_frame = x_pooled.max(dim=2)[0]  # (B, T, embed_dim)

        # Add temporal positional encoding
        if T <= self.temporal_pos.shape[1]:
            x_frame = x_frame + self.temporal_pos[:, :T, :]
        else:
            x_frame = x_frame + self.temporal_pos.repeat(1, T // self.temporal_pos.shape[1] + 1, 1)[:, :T, :]

        # Inter-frame Temporal Mamba
        x_temporal = self.temporal_mamba(x_frame)  # (B, T, embed_dim)

        # Final normalization
        x_temporal = self.norm(x_temporal)

        # Global temporal pooling (mean over frames)
        x_global = x_temporal.mean(dim=1)  # (B, embed_dim)

        # Output head
        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](x_global))
            y = torch.stack(y, dim=1)
        else:
            y = self.output(x_global)

        return y

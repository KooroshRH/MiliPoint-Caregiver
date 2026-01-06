"""
PointMamba: A Simple State Space Model for Point Cloud Analysis
Based on: https://arxiv.org/abs/2402.10739 (NeurIPS 2024)

Key innovations:
- Space-filling curve serialization (Hilbert/Trans-Hilbert)
- Non-hierarchical Mamba encoder
- Linear complexity global modeling

Requires: mamba-ssm, causal-conv1d
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MLP, knn, global_mean_pool

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. PointMamba will not be available.")


def hilbert_curve_indices(n_points):
    """
    Generate Hilbert curve indices for ordering points.
    Simplified 3D Hilbert curve approximation using z-order (Morton code) with bit interleaving.

    For true Hilbert curve, points should be normalized to [0, 1] range first.
    """
    # We'll compute this dynamically based on point positions
    return None  # Placeholder - actual ordering done in forward pass


def z_order_key(x, y, z, bits=10):
    """
    Compute Z-order (Morton code) for 3D point.
    Interleaves bits of x, y, z coordinates.
    """
    def spread_bits(v):
        v = v & 0x3ff  # 10 bits
        v = (v | (v << 16)) & 0x30000ff
        v = (v | (v << 8)) & 0x300f00f
        v = (v | (v << 4)) & 0x30c30c3
        v = (v | (v << 2)) & 0x9249249
        return v

    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def get_space_filling_order(xyz, order_type='hilbert'):
    """
    Get point indices sorted by space-filling curve order.

    Args:
        xyz: (N, 3) point coordinates
        order_type: 'hilbert' or 'trans_hilbert' or 'z_order'

    Returns:
        indices: (N,) sorted indices
    """
    # Normalize to [0, 1023] range for 10-bit Morton code
    xyz_min = xyz.min(dim=0, keepdim=True)[0]
    xyz_max = xyz.max(dim=0, keepdim=True)[0]
    xyz_range = xyz_max - xyz_min + 1e-6
    xyz_norm = ((xyz - xyz_min) / xyz_range * 1023).long().clamp(0, 1023)

    if order_type == 'hilbert' or order_type == 'z_order':
        # Z-order (Morton code) - approximates Hilbert
        keys = z_order_key(xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2])
    elif order_type == 'trans_hilbert':
        # Transposed - swap axes for different traversal
        keys = z_order_key(xyz_norm[:, 2], xyz_norm[:, 1], xyz_norm[:, 0])
    else:
        raise ValueError(f"Unknown order type: {order_type}")

    return torch.argsort(keys)


class PointPatchEmbed(nn.Module):
    """
    Point patch embedding using KNN grouping and mini-PointNet.
    """
    def __init__(self, in_channels=3, embed_dim=384, num_patches=64, k_neighbors=32):
        super().__init__()
        self.num_patches = num_patches
        self.k_neighbors = k_neighbors
        self.embed_dim = embed_dim

        # Mini-PointNet for patch embedding
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, embed_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        self.act = nn.GELU()

    def forward(self, xyz, features, batch):
        """
        Args:
            xyz: (N, 3) point positions
            features: (N, C) point features (can be same as xyz)
            batch: (N,) batch indices
        Returns:
            patch_tokens: (B, num_patches, embed_dim)
            patch_centers: (B, num_patches, 3)
        """
        B = batch.max().item() + 1
        device = xyz.device

        all_tokens = []
        all_centers = []

        for b in range(B):
            mask = (batch == b)
            xyz_b = xyz[mask]  # (N_b, 3)
            feat_b = features[mask]  # (N_b, C)
            n_points = xyz_b.shape[0]

            # Select patch centers using FPS-like uniform sampling
            if n_points <= self.num_patches:
                # If fewer points than patches, use all points
                center_idx = torch.arange(n_points, device=device)
                actual_patches = n_points
            else:
                # Uniform sampling for patch centers
                step = n_points / self.num_patches
                center_idx = (torch.arange(self.num_patches, device=device) * step).long()
                actual_patches = self.num_patches

            centers = xyz_b[center_idx]  # (P, 3)

            # KNN for each center
            k = min(self.k_neighbors, n_points)

            # Find k nearest neighbors for each center
            # Using cdist for simplicity
            dists = torch.cdist(centers, xyz_b)  # (P, N_b)
            _, knn_idx = dists.topk(k, dim=1, largest=False)  # (P, k)

            # Gather neighbor features
            neighbors = feat_b[knn_idx]  # (P, k, C)
            neighbor_xyz = xyz_b[knn_idx]  # (P, k, 3)

            # Normalize by center
            neighbor_xyz = neighbor_xyz - centers.unsqueeze(1)  # (P, k, 3)

            # Concatenate if features != xyz
            if features.shape[1] != 3:
                patch_feat = torch.cat([neighbor_xyz, neighbors], dim=-1)  # (P, k, 3+C)
            else:
                patch_feat = neighbor_xyz  # (P, k, 3)

            # Mini-PointNet: (P, k, C) -> (P, C, k) -> conv -> (P, embed_dim)
            patch_feat = patch_feat.transpose(1, 2)  # (P, C, k)
            patch_feat = self.act(self.bn1(self.conv1(patch_feat)))
            patch_feat = self.act(self.bn2(self.conv2(patch_feat)))
            patch_feat = self.bn3(self.conv3(patch_feat))  # (P, embed_dim, k)
            patch_feat = patch_feat.max(dim=2)[0]  # (P, embed_dim)

            # Pad if needed
            if actual_patches < self.num_patches:
                pad_size = self.num_patches - actual_patches
                patch_feat = torch.cat([patch_feat, patch_feat[:pad_size]], dim=0)
                centers = torch.cat([centers, centers[:pad_size]], dim=0)

            all_tokens.append(patch_feat)
            all_centers.append(centers)

        patch_tokens = torch.stack(all_tokens, dim=0)  # (B, num_patches, embed_dim)
        patch_centers = torch.stack(all_centers, dim=0)  # (B, num_patches, 3)

        return patch_tokens, patch_centers


class MambaBlock(nn.Module):
    """
    Mamba block with residual connection.
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
            x: (B, L, D) sequence
        Returns:
            (B, L, D) transformed sequence
        """
        return x + self.dropout(self.mamba(self.norm(x)))


class OrderIndicator(nn.Module):
    """
    Learnable scale-shift parameters to distinguish different serialization orders.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.scale + self.shift


class PointMamba(nn.Module):
    """
    PointMamba for point cloud classification.

    Input format: (B, N, 3) for vanilla version (XYZ only)
    Output format: (B, num_classes) for classification

    Architecture:
    - Point patch embedding via KNN + mini-PointNet
    - Bidirectional Mamba encoding (Hilbert + Trans-Hilbert)
    - 12 Mamba blocks, 384 hidden dimensions
    """
    def __init__(
        self,
        info=None,
        in_channels=3,
        embed_dim=384,
        depth=12,
        num_patches=64,
        k_neighbors=32,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required for PointMamba. Install with: pip install mamba-ssm")

        self.num_classes = info['num_classes']
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Patch embedding
        self.patch_embed = PointPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_patches=num_patches,
            k_neighbors=k_neighbors
        )

        # Order indicators for Hilbert and Trans-Hilbert
        self.order_indicator_h = OrderIndicator(embed_dim)
        self.order_indicator_th = OrderIndicator(embed_dim)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mamba encoder blocks
        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim, d_state, d_conv, expand, dropout)
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        if self.num_classes is None:
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([embed_dim, 256, 64, 3], dropout=0.5, norm=None)
            self.output = nn.ModuleDict(point_branches)
        else:
            self.output = MLP([embed_dim, 256, 64, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        """
        Args:
            data: (B, N, C) point cloud, C >= 3
        Returns:
            (B, num_classes) classification logits
        """
        B, N, C = data.shape
        device = data.device

        # Extract XYZ
        xyz = data[:, :, :3]

        # Reshape for patch embedding
        xyz_flat = xyz.reshape(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        # Patch embedding
        patch_tokens, patch_centers = self.patch_embed(xyz_flat, xyz_flat, batch)
        # patch_tokens: (B, num_patches, embed_dim)
        # patch_centers: (B, num_patches, 3)

        # Apply space-filling curve ordering for each sample
        tokens_h_list = []
        tokens_th_list = []

        for b in range(B):
            centers_b = patch_centers[b]  # (num_patches, 3)
            tokens_b = patch_tokens[b]  # (num_patches, embed_dim)

            # Hilbert order
            order_h = get_space_filling_order(centers_b, 'hilbert')
            tokens_h = tokens_b[order_h]
            tokens_h = self.order_indicator_h(tokens_h)
            tokens_h_list.append(tokens_h)

            # Trans-Hilbert order
            order_th = get_space_filling_order(centers_b, 'trans_hilbert')
            tokens_th = tokens_b[order_th]
            tokens_th = self.order_indicator_th(tokens_th)
            tokens_th_list.append(tokens_th)

        tokens_h = torch.stack(tokens_h_list, dim=0)  # (B, P, D)
        tokens_th = torch.stack(tokens_th_list, dim=0)  # (B, P, D)

        # Add positional embedding
        tokens_h = tokens_h + self.pos_embed
        tokens_th = tokens_th + self.pos_embed

        # Concatenate bidirectional sequences
        # (B, 2*P, D)
        tokens = torch.cat([tokens_h, tokens_th], dim=1)

        # Mamba encoder
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Global pooling (mean over sequence)
        x = tokens.mean(dim=1)  # (B, embed_dim)

        # Output head
        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.output(x)

        return y

"""
PointMamba with Auxiliary Features Support
Based on: https://arxiv.org/abs/2402.10739 (NeurIPS 2024)

Modified to support full auxiliary data (XYZ + auxiliary channels)
Input: (B, N, 7) where channels are [X, Y, Z, Zone, Doppler, SNR, Density]

Requires: mamba-ssm, causal-conv1d
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MLP

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


def z_order_key(x, y, z, bits=10):
    """Compute Z-order (Morton code) for 3D point."""
    def spread_bits(v):
        v = v & 0x3ff
        v = (v | (v << 16)) & 0x30000ff
        v = (v | (v << 8)) & 0x300f00f
        v = (v | (v << 4)) & 0x30c30c3
        v = (v | (v << 2)) & 0x9249249
        return v
    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def get_space_filling_order(xyz, order_type='hilbert'):
    """Get point indices sorted by space-filling curve order."""
    xyz_min = xyz.min(dim=0, keepdim=True)[0]
    xyz_max = xyz.max(dim=0, keepdim=True)[0]
    xyz_range = xyz_max - xyz_min + 1e-6
    xyz_norm = ((xyz - xyz_min) / xyz_range * 1023).long().clamp(0, 1023)

    if order_type == 'hilbert' or order_type == 'z_order':
        keys = z_order_key(xyz_norm[:, 0], xyz_norm[:, 1], xyz_norm[:, 2])
    elif order_type == 'trans_hilbert':
        keys = z_order_key(xyz_norm[:, 2], xyz_norm[:, 1], xyz_norm[:, 0])
    else:
        raise ValueError(f"Unknown order type: {order_type}")

    return torch.argsort(keys)


class PointPatchEmbedAux(nn.Module):
    """
    Point patch embedding with auxiliary features using KNN grouping and mini-PointNet.
    """
    def __init__(self, in_channels=7, embed_dim=384, num_patches=64, k_neighbors=32):
        super().__init__()
        self.num_patches = num_patches
        self.k_neighbors = k_neighbors
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # Mini-PointNet for patch embedding
        # Input: relative xyz (3) + auxiliary features (in_channels - 3) = in_channels
        # But we also keep relative xyz, so total = 3 + in_channels
        patch_in_dim = 3 + in_channels  # relative xyz + full features

        self.conv1 = nn.Conv1d(patch_in_dim, 64, 1)
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
            features: (N, C) point features (full data including xyz + aux)
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
            xyz_b = xyz[mask]
            feat_b = features[mask]
            n_points = xyz_b.shape[0]

            # Select patch centers
            if n_points <= self.num_patches:
                center_idx = torch.arange(n_points, device=device)
                actual_patches = n_points
            else:
                step = n_points / self.num_patches
                center_idx = (torch.arange(self.num_patches, device=device) * step).long()
                actual_patches = self.num_patches

            centers = xyz_b[center_idx]

            # KNN for each center
            k = min(self.k_neighbors, n_points)
            dists = torch.cdist(centers, xyz_b)
            _, knn_idx = dists.topk(k, dim=1, largest=False)

            # Gather neighbor features
            neighbors = feat_b[knn_idx]  # (P, k, C)
            neighbor_xyz = xyz_b[knn_idx]  # (P, k, 3)

            # Relative xyz normalized by center
            rel_xyz = neighbor_xyz - centers.unsqueeze(1)  # (P, k, 3)

            # Concatenate relative xyz with full features
            patch_feat = torch.cat([rel_xyz, neighbors], dim=-1)  # (P, k, 3+C)

            # Mini-PointNet
            patch_feat = patch_feat.transpose(1, 2)  # (P, 3+C, k)
            patch_feat = self.act(self.bn1(self.conv1(patch_feat)))
            patch_feat = self.act(self.bn2(self.conv2(patch_feat)))
            patch_feat = self.bn3(self.conv3(patch_feat))
            patch_feat = patch_feat.max(dim=2)[0]  # (P, embed_dim)

            # Pad if needed
            if actual_patches < self.num_patches:
                pad_size = self.num_patches - actual_patches
                patch_feat = torch.cat([patch_feat, patch_feat[:pad_size]], dim=0)
                centers = torch.cat([centers, centers[:pad_size]], dim=0)

            all_tokens.append(patch_feat)
            all_centers.append(centers)

        patch_tokens = torch.stack(all_tokens, dim=0)
        patch_centers = torch.stack(all_centers, dim=0)

        return patch_tokens, patch_centers


class MambaBlock(nn.Module):
    """Mamba block with residual connection."""
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
        return x + self.dropout(self.mamba(self.norm(x)))


class OrderIndicator(nn.Module):
    """Learnable scale-shift for different serialization orders."""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.scale + self.shift


class PointMamba_Aux(nn.Module):
    """
    PointMamba with auxiliary feature support.

    Input format: (B, N, C) where C = 7 (XYZ + 4 auxiliary channels)
    Output format: (B, num_classes) for classification

    Architecture:
    - Point patch embedding with auxiliary features via KNN + mini-PointNet
    - Bidirectional Mamba encoding (Hilbert + Trans-Hilbert)
    - 12 Mamba blocks, 384 hidden dimensions
    """
    def __init__(
        self,
        info=None,
        in_channels=7,
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
            raise ImportError("mamba-ssm is required for PointMamba_Aux. Install with: pip install mamba-ssm")

        self.num_classes = info['num_classes']
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Patch embedding with auxiliary features
        self.patch_embed = PointPatchEmbedAux(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_patches=num_patches,
            k_neighbors=k_neighbors
        )

        # Order indicators
        self.order_indicator_h = OrderIndicator(embed_dim)
        self.order_indicator_th = OrderIndicator(embed_dim)

        # Positional embedding
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
            data: (B, N, C) point cloud with auxiliary features
        Returns:
            (B, num_classes) classification logits
        """
        B, N, C = data.shape
        device = data.device

        # Extract XYZ for positions
        xyz = data[:, :, :3]

        # Reshape for patch embedding
        xyz_flat = xyz.reshape(B * N, 3)
        data_flat = data.reshape(B * N, C)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        # Patch embedding (uses both xyz and auxiliary features)
        patch_tokens, patch_centers = self.patch_embed(xyz_flat, data_flat, batch)

        # Apply space-filling curve ordering
        tokens_h_list = []
        tokens_th_list = []

        for b in range(B):
            centers_b = patch_centers[b]
            tokens_b = patch_tokens[b]

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

        tokens_h = torch.stack(tokens_h_list, dim=0)
        tokens_th = torch.stack(tokens_th_list, dim=0)

        # Add positional embedding
        tokens_h = tokens_h + self.pos_embed
        tokens_th = tokens_th + self.pos_embed

        # Concatenate bidirectional sequences
        tokens = torch.cat([tokens_h, tokens_th], dim=1)

        # Mamba encoder
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Global pooling
        x = tokens.mean(dim=1)

        # Output head
        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.output(x)

        return y

"""
Point Transformer V3 with Auxiliary Features Support
Based on: https://arxiv.org/abs/2312.10035 (CVPR 2024 Oral)

Modified to support full auxiliary data (XYZ + auxiliary channels)
Input: (B, N, 7) where channels are [X, Y, Z, Zone, Doppler, SNR, Density]
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MLP, fps, global_mean_pool, knn
from torch_geometric.utils import scatter


class SerializedAttention(nn.Module):
    """
    Serialized patch attention for PTv3.
    Groups points into patches and computes attention within each patch.
    """
    def __init__(self, channels, num_heads=8, patch_size=64, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, batch):
        """
        Args:
            x: (N, C) point features
            batch: (N,) batch indices
        Returns:
            (N, C) transformed features
        """
        B = batch.max().item() + 1
        N = x.shape[0]

        # Compute QKV
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)  # (3, N, heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Process each batch separately with patch-based attention
        out_list = []

        for b in range(B):
            mask = (batch == b)
            n_points = mask.sum().item()

            if n_points == 0:
                continue

            q_b = q[mask]  # (n_points, heads, head_dim)
            k_b = k[mask]
            v_b = v[mask]

            # Split into patches
            n_patches = max(1, (n_points + self.patch_size - 1) // self.patch_size)
            actual_patch_size = (n_points + n_patches - 1) // n_patches

            # Pad if needed
            pad_size = n_patches * actual_patch_size - n_points
            if pad_size > 0:
                q_b = torch.cat([q_b, q_b[:pad_size]], dim=0)
                k_b = torch.cat([k_b, k_b[:pad_size]], dim=0)
                v_b = torch.cat([v_b, v_b[:pad_size]], dim=0)

            # Reshape into patches
            q_b = q_b.reshape(n_patches, actual_patch_size, self.num_heads, self.head_dim)
            k_b = k_b.reshape(n_patches, actual_patch_size, self.num_heads, self.head_dim)
            v_b = v_b.reshape(n_patches, actual_patch_size, self.num_heads, self.head_dim)

            # Attention within patches
            q_b = q_b.permute(0, 2, 1, 3)
            k_b = k_b.permute(0, 2, 1, 3)
            v_b = v_b.permute(0, 2, 1, 3)

            attn = (q_b @ k_b.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out_b = (attn @ v_b)
            out_b = out_b.permute(0, 2, 1, 3).reshape(-1, self.channels)

            # Remove padding
            out_b = out_b[:n_points]
            out_list.append(out_b)

        out = torch.cat(out_list, dim=0)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class PTv3Block(nn.Module):
    """
    Point Transformer V3 block with pre-norm structure.
    """
    def __init__(self, channels, num_heads=8, patch_size=64, mlp_ratio=4, drop=0.0):
        super().__init__()

        # xCPE: enhanced conditional positional encoding
        self.cpe = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

        # Pre-norm attention
        self.norm1 = nn.LayerNorm(channels)
        self.attn = SerializedAttention(
            channels, num_heads=num_heads, patch_size=patch_size,
            attn_drop=drop, proj_drop=drop
        )

        # Pre-norm MLP
        self.norm2 = nn.LayerNorm(channels)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, channels),
            nn.Dropout(drop)
        )

    def forward(self, x, batch):
        x = x + self.cpe(x)
        x = x + self.attn(self.norm1(x), batch)
        x = x + self.mlp(self.norm2(x))
        return x


class PTv3Stage(nn.Module):
    """
    A stage in PTv3 encoder.
    """
    def __init__(self, in_channels, out_channels, num_blocks, num_heads, patch_size, drop=0.0):
        super().__init__()

        self.proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        self.blocks = nn.ModuleList([
            PTv3Block(out_channels, num_heads=num_heads, patch_size=patch_size, drop=drop)
            for _ in range(num_blocks)
        ])

    def forward(self, x, pos, batch):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, batch)
        return x, pos, batch


class PTv3DownSample(nn.Module):
    """
    Downsampling layer using FPS + KNN pooling
    """
    def __init__(self, in_channels, out_channels, ratio=0.5, k=16):
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x, pos, batch):
        # FPS sampling
        idx = fps(pos, batch, ratio=self.ratio)
        new_pos = pos[idx]
        new_batch = batch[idx]

        # KNN from original to sampled
        edge_index = knn(pos, new_pos, self.k, batch, new_batch)

        # Transform features
        x = self.mlp(x)

        # Max pool from neighbors
        new_x = scatter(x[edge_index[1]], edge_index[0], dim=0,
                       dim_size=new_pos.size(0), reduce='max')

        return new_x, new_pos, new_batch


class PointTransformerV3_Aux(nn.Module):
    """
    Point Transformer V3 with auxiliary feature support.

    Input format: (B, N, C) where C = 7 (XYZ + 4 auxiliary channels)
    Output format: (B, num_classes) for classification

    Architecture follows PTv3 paper with 4 encoder stages:
    - Channels: [64, 128, 256, 512]
    - Depths: [2, 2, 6, 2]
    - Heads: [4, 8, 16, 32]
    """
    def __init__(
        self,
        info=None,
        in_channels=7,
        channels=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=64,
        k=16,
        drop=0.1
    ):
        super().__init__()
        self.num_classes = info['num_classes']
        self.in_channels = in_channels

        # Input embedding - takes all channels including auxiliary
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.GELU()
        )

        # Encoder stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(channels)):
            in_ch = channels[i-1] if i > 0 else channels[0]
            out_ch = channels[i]

            self.stages.append(
                PTv3Stage(
                    in_ch, out_ch,
                    num_blocks=depths[i],
                    num_heads=num_heads[i],
                    patch_size=patch_size,
                    drop=drop
                )
            )

            if i < len(channels) - 1:
                self.downsamples.append(
                    PTv3DownSample(out_ch, out_ch, ratio=0.5, k=k)
                )

        # Classification head
        if self.num_classes is None:
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([channels[-1], 256, 64, 3], dropout=0.5, norm=None)
            self.output = nn.ModuleDict(point_branches)
        else:
            self.output = MLP([channels[-1], 256, 64, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        batchsize = data.shape[0]
        npoints = data.shape[1]

        # Reshape to (B*N, C) - use all channels
        x = data.reshape(batchsize * npoints, self.in_channels)

        # Extract XYZ for position (first 3 channels)
        pos = x[:, :3].clone()

        batch = torch.arange(batchsize, device=x.device).repeat_interleave(npoints)

        # Input projection (uses all channels including auxiliary)
        x = self.input_proj(x)

        # Encoder stages
        for i, stage in enumerate(self.stages):
            x, pos, batch = stage(x, pos, batch)
            if i < len(self.downsamples):
                x, pos, batch = self.downsamples[i](x, pos, batch)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Output head
        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.output(x)

        return y

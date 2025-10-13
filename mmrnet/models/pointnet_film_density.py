import torch
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class ZoneFiLM(nn.Module):
    def __init__(self, feat_dim, zone_dim=6, embed_dim=16):
        super().__init__()
        self.zone_embed = nn.Linear(zone_dim, embed_dim)   # embed zone one-hot
        self.gamma_gen = nn.Linear(embed_dim, feat_dim)    # generate scale
        self.beta_gen  = nn.Linear(embed_dim, feat_dim)    # generate shift

    def forward(self, f, zone_onehot):
        """
        f: (B*N, feat_dim)  - per-point features
        zone_onehot: (B*N, zone_dim) - one-hot zone
        """
        e = self.zone_embed(zone_onehot)       # (B*N, embed_dim)
        gamma = self.gamma_gen(e)              # (B*N, feat_dim)
        beta = self.beta_gen(e)                # (B*N, feat_dim)
        return gamma * f + beta

class PointNetWithFiLMDensity(torch.nn.Module):
    def __init__(self, info, zone_dim=6):
        super().__init__()
        self.num_classes = info['num_classes']
        self.zone_dim = zone_dim

        # Updated MLP layers to handle 4D node features + 3D position features = 7D input
        # PointNetConv concatenates node features (4D) with position features (3D)
        self.sa1_module = SAModule(0.5, 0.2, MLP([4 + 3, 64, 64, 128]))  # 4+3=7 instead of 3+3=6
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))  # pos is always 3D
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))  # pos is always 3D

        # FiLM module for zone conditioning
        self.zone_film = ZoneFiLM(feat_dim=128, zone_dim=zone_dim)

        if self.num_classes is None:
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([1024, 512, 256, 3])
            self.mlp = torch.nn.ModuleDict(point_branches)
        else:
            self.mlp = MLP([1024, 512, 256, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        # data: (B, N, 5) where dimensions are [x, y, z, zone_index, density]

        batchsize, npoints, _ = data.shape

        # Extract components
        xyz = data[:, :, :3]  # (B, N, 3) - x, y, z coordinates
        zone_indices = data[:, :, 3].long()  # (B, N) - zone indices
        density = data[:, :, 4:5]  # (B, N, 1) - density values

        # Concatenate xyz with density to create 4D input for backbone
        xyzd = torch.cat([xyz, density], dim=-1)  # (B, N, 4)

        # Convert zone indices to one-hot encoding
        zone_onehot = torch.zeros(batchsize, npoints, self.zone_dim, device=data.device)
        zone_onehot.scatter_(2, zone_indices.unsqueeze(-1), 1)  # (B, N, zone_dim)

        # Reshape for processing
        x = xyzd.reshape((batchsize * npoints, 4))  # 4D input instead of 3D
        pos = xyz.reshape((batchsize * npoints, 3))  # Position still 3D for spatial operations
        zone_flat = zone_onehot.reshape((batchsize * npoints, self.zone_dim))  # flatten

        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
        sa0_out = (x, pos, batch)  # x is 4D, pos is 3D

        # SA1 - we need to manually perform FPS to get indices for zone propagation
        # Get FPS indices (same as in SA1)
        sa1_idx = fps(pos, batch, ratio=0.5)  # Same ratio as sa1_module

        # Apply zone information using FPS indices
        zone_sa1_flat = zone_flat[sa1_idx]  # Direct indexing using FPS indices

        # Now run SA1 normally
        x, pos, batch = self.sa1_module(*sa0_out)

        # Apply FiLM on per-point features (dimensions now match!)
        x = self.zone_film(x, zone_sa1_flat)

        # SA2 + SA3 as usual
        sa2_out = self.sa2_module(x, pos, batch)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # Classifier head
        if self.num_classes is None:
            y = [self.mlp[f'branch_{i}'](x) for i in range(self.num_points)]
            y = torch.stack(y, dim=1)
        else:
            y = self.mlp(x)
        return y
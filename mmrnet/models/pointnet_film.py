import torch
import torch.nn as nn
from .pointnet import PointNet

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

class PointNetWithFiLM(PointNet):
    def __init__(self, info, zone_dim=6):
        super().__init__(info)
        self.zone_film = ZoneFiLM(feat_dim=128, zone_dim=zone_dim)  
        self.zone_dim = zone_dim
        # 128 = output dim of sa1_module in your code

    def forward(self, data):
        # data: (B, N, 4) where last dimension is zone index
        
        batchsize, npoints, _ = data.shape
        
        # Extract xyz coordinates and zone indices
        xyz = data[:, :, :3]  # (B, N, 3)
        zone_indices = data[:, :, 3].long()  # (B, N) - zone indices
        
        # Convert zone indices to one-hot encoding
        zone_onehot = torch.zeros(batchsize, npoints, self.zone_dim, device=data.device)
        zone_onehot.scatter_(2, zone_indices.unsqueeze(-1), 1)  # (B, N, zone_dim)
        
        # Reshape for processing
        x = xyz.reshape((batchsize * npoints, 3))
        zone_flat = zone_onehot.reshape((batchsize * npoints, self.zone_dim))  # flatten

        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
        sa0_out = (x, x, batch)

        # SA1 - we need to manually perform FPS to get indices for zone propagation
        from torch_geometric.nn import fps
        
        # Get FPS indices (same as in SA1)
        sa1_idx = fps(x, batch, ratio=0.5)  # Same ratio as sa1_module
        
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

# Modified from pointnet.py to support full auxiliary data
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
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

class PointNet_Aux(torch.nn.Module):
    def __init__(self, info=None, in_channels=7):
        super().__init__()
        self.num_classes = info['num_classes']
        self.in_channels = in_channels

        # Input channels account for both `pos` and node features.
        # For the first SAModule, we have in_channels input features + 3 position features
        # The PointNetConv will concatenate features from source and destination, so input to MLP is (in_channels + 3)
        self.sa1_module = SAModule(0.5, 0.2, MLP([in_channels + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        if self.num_classes is None:
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([1024, 512, 256, 3])
            self.mlp = torch.nn.ModuleDict(point_branches)
        else:
            self.mlp = MLP([1024, 512, 256, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        batchsize = data.shape[0]
        npoints = data.shape[1]

        # Use all channels (xyz + auxiliary data)
        x = data.reshape((batchsize * npoints, self.in_channels))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)

        # Extract position from first 3 channels (xyz coordinates)
        pos = x[:, :3]

        # Initial input: use full feature vector (including xyz and aux)
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.mlp[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.mlp(x)
        return y


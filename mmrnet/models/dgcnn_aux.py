# Modified from dgcnn.py to support full auxiliary data
# V2: accepts (point_cloud, frame_signals) tuple from new data pipeline.
# point_cloud   : (B, T, N, 6)  [X, Y, Z, Doppler, SNR, Density]
# frame_signals : (B, T, 9)     [acc(3), gyro(3), BLE(3)]
# Frame signals are broadcast to every point and concatenated → 15D input per point.
# Temporal frames are flattened into the point dimension (B*T*N points per batch).
import torch
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from torch import nn

class DGCNN_Aux(torch.nn.Module):
    def __init__(self, k=30, aggr='max', info=None, in_channels=15):
        super().__init__()
        self.num_classes = info['num_classes']
        self.in_channels = in_channels
        self.conv = nn.ModuleList([])
        n = in_channels
        conv_layer=(32, 32, 32)
        dense_layer=(1024, 1024, 256, 128)

        for layer in conv_layer:
            edgeconv = DynamicEdgeConv(MLP([n * 2, layer, layer, layer]), k, aggr)
            self.conv.append(edgeconv)
            n = layer

        self.lin1 = MLP([sum(conv_layer), dense_layer[0]])

        if self.num_classes is None:    # keypoint dataset
            num_points = info['num_keypoints']
            self.num_points = num_points
            point_branches = {}
            for i in range(num_points):
                point_branches[f'branch_{i}'] = MLP([*dense_layer, 3], dropout=0.5, norm=None)
            self.output = torch.nn.ModuleDict(point_branches)
        else:                           # identification or action
            self.output = MLP([*dense_layer, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        point_cloud, frame_signals = data
        # point_cloud   : (B, T, N, 6)
        # frame_signals : (B, T, 9)
        B, T, N, _ = point_cloud.shape

        # Broadcast frame signals to every point in the frame: (B, T, 9) -> (B, T, N, 9)
        fs = frame_signals.unsqueeze(2).expand(-1, -1, N, -1)  # (B, T, N, 9)

        # Concatenate point-level and frame-level features: (B, T, N, 15)
        x = torch.cat([point_cloud, fs], dim=-1)

        # Flatten temporal frames into point dimension: (B*T*N, 15)
        x = x.reshape(B * T * N, self.in_channels)
        batch = torch.arange(B * T, device=x.device).repeat_interleave(N)

        xs = []
        for conv in self.conv:
            x = conv(x, batch)
            xs.append(x)

        x4 = self.lin1(torch.cat(xs, dim=1))
        x5 = global_max_pool(x4, batch)          # (B*T, dense_layer[0])

        # Mean pool over T frames to get one vector per batch item
        x5 = x5.view(B, T, -1).mean(dim=1)       # (B, dense_layer[0])

        if self.num_classes is None:    # keypoint dataset
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](x5))
            y = torch.stack(y, dim=1)
        else:                           # identification or action
            y = self.output(x5)
        return y



# dgcnn_aux_t.py — DGCNN with auxiliary features (15D concat) + Temporal Transformer
#
# Same as dgcnn_aux but replaces mean-pool over T frames with a Transformer encoder.
# This isolates the contribution of the temporal transformer independent of any
# point-level FiLM or frame-level cross-attention conditioning.
#
# Input: (point_cloud, frame_signals) tuple
#   point_cloud   : (B, T, N, 6)  [X, Y, Z, Doppler, SNR, Density]
#   frame_signals : (B, T, 9)     [acc(3), gyro(3), BLE(3)]
# Frame signals are broadcast to every point and concatenated → 15D per point.
# Temporal frames processed independently by EdgeConv, then Transformer across T.

import torch
import torch.nn as nn
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool


class DGCNN_Aux_T(nn.Module):
    def __init__(self,
                 k=30,
                 aggr='max',
                 info=None,
                 in_channels=15,
                 temporal_layers=1,
                 temporal_heads=4,
                 use_temporal_pos_embed=True):
        super().__init__()
        self.num_classes = info['num_classes']
        self.in_channels = in_channels

        conv_layer = (32, 32, 32)
        dense_layer = (1024, 1024, 256, 128)
        self.temporal_dim = dense_layer[0]

        # Spatial backbone: EdgeConv stack (same as dgcnn_aux)
        self.conv = nn.ModuleList()
        n = in_channels
        for layer in conv_layer:
            self.conv.append(DynamicEdgeConv(MLP([n * 2, layer, layer, layer]), k, aggr))
            n = layer

        self.lin1 = MLP([sum(conv_layer), self.temporal_dim])

        # Temporal encoder: Transformer across T frames
        if temporal_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.temporal_dim,
                nhead=temporal_heads,
                dim_feedforward=max(512, self.temporal_dim * 2),
                dropout=0.1,
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
            if use_temporal_pos_embed:
                self.time_pos_embed = nn.Parameter(torch.randn(1, 64, self.temporal_dim) * 0.02)
            else:
                self.time_pos_embed = None
        else:
            self.temporal_encoder = None
            self.time_pos_embed = None

        # Classification head
        self.output = MLP([*dense_layer, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        point_cloud, frame_signals = data
        # point_cloud   : (B, T, N, 6)
        # frame_signals : (B, T, 9)
        B, T, N, _ = point_cloud.shape
        device = point_cloud.device

        # Broadcast frame signals to every point: (B, T, N, 9)
        fs = frame_signals.unsqueeze(2).expand(-1, -1, N, -1)

        # Concatenate → (B, T, N, 15), flatten → (B*T*N, 15)
        x = torch.cat([point_cloud, fs], dim=-1).reshape(B * T * N, self.in_channels)
        frame_idx = torch.arange(B * T, device=device).repeat_interleave(N)

        # EdgeConv stack — each frame processed independently
        xs = []
        for conv in self.conv:
            x = conv(x, frame_idx)
            xs.append(x)

        x = self.lin1(torch.cat(xs, dim=1))         # (B*T*N, temporal_dim)
        E = global_max_pool(x, frame_idx)            # (B*T, temporal_dim)
        E = E.view(B, T, -1)                         # (B, T, temporal_dim)

        # Temporal transformer
        if self.temporal_encoder is not None:
            if self.time_pos_embed is not None:
                max_pos = self.time_pos_embed.size(1)
                if T <= max_pos:
                    pos = self.time_pos_embed[:, :T, :].expand(B, -1, -1)
                else:
                    reps = (T + max_pos - 1) // max_pos
                    pos = self.time_pos_embed.repeat(1, reps, 1)[:, :T, :].expand(B, -1, -1)
                E = E + pos
            E = self.temporal_encoder(E)             # (B, T, temporal_dim)

        feat = E.mean(dim=1)                         # (B, temporal_dim)
        return self.output(feat)                     # (B, num_classes)

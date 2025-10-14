# point_transformer_with_cross_attn.py
import torch
import torch.nn as nn
from torch.nn import Linear as Lin
from torch_geometric.nn import MLP, PointTransformerConv, fps, global_mean_pool, knn_graph
from torch_geometric.utils import scatter


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)
        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, plain_last=False)
        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """
    Downsample via FPS + kNN pooling. Also allow optional aux pooling so aux tokens align with features.
    """
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch, aux=None):
        """
        x: (num_points, Cx)
        pos: (num_points, 3)
        batch: (num_points,)
        aux: (num_points, Caux) or None
        returns: out_features (num_clusters, Cout), sub_pos (num_clusters,3), sub_batch (num_clusters,),
                 aux_out (num_clusters, Caux) if aux provided else None, id_clusters (indices into original)
        """
        # FPS sampling (per-batch)
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # get batch indices for sampled
        sub_batch = batch[id_clusters] if batch is not None else None

        # knn from all points -> sampled centers (edges: (2, E) with [center_idx, neighbor_idx])
        id_k_neighbor = knn_graph(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)

        # transform features
        x_trans = self.mlp(x)

        # pool (max) neighbor features into cluster centers
        x_out = scatter(x_trans[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')

        aux_out = None
        if aux is not None:
            # For aux, we can either take the aux of sampled point directly, or pool neighbors similarly.
            # We'll pool aux values (mean) across neighbors for stability.
            aux_neighbor_vals = aux[id_k_neighbor[1]]  # (E, Caux)
            aux_out = scatter(aux_neighbor_vals, id_k_neighbor[0], dim=0,
                              dim_size=id_clusters.size(0), reduce='mean')

        sub_pos = pos[id_clusters]
        return x_out, sub_pos, sub_batch, aux_out, id_clusters


class AuxEncoder(nn.Module):
    """
    Encodes per-point aux features into per-stage aux tokens (one token projection per stage/dim).
    We create a small shared trunk, then per-stage projection heads to produce aux tokens that match
    geometry dims at each stage.
    """
    def __init__(self, aux_dim, stage_dims):
        """
        aux_dim: number of aux channels per point
        stage_dims: list of ints; token dim for each stage (e.g. [d0, d1, d2,...])
        """
        super().__init__()
        self.aux_dim = aux_dim
        self.trunk = MLP([aux_dim, 64, 64], plain_last=False)
        self.heads = nn.ModuleList([MLP([64, d], plain_last=True, norm=None) for d in stage_dims])

    def forward(self, aux):
        """
        aux: (num_points, aux_dim)
        returns: list of aux tokens per stage: [ (num_points, d0), (num_points, d1), ... ]
        """
        h = self.trunk(aux)
        outs = [head(h) for head in self.heads]
        return outs


class PointCrossAttention(nn.Module):
    """
    Lightweight point-wise cross-attention (cheap & effective).
    For each point i, compute:
      q = Wq * geom_i
      k = Wk * aux_i
      v = Wv * aux_i
      gate = sigmoid( (q * k).sum(dim=-1) / sqrt(dim) )
      out_i = Wout( v * gate.unsqueeze(-1) )
    This is O(N) and keeps point alignment; it's more expressive than simple concat/FiLM while being cheap.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.wq = Lin(dim, dim)
        self.wk = Lin(dim, dim)
        self.wv = Lin(dim, dim)
        self.out = Lin(dim, dim)
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, geom, aux):
        """
        geom: (num_tokens, dim)
        aux:  (num_tokens, dim)  (tokens aligned 1:1)
        returns: fused (num_tokens, dim)
        """
        q = self.wq(geom)          # (N, d)
        k = self.wk(aux)           # (N, d)
        v = self.wv(aux)           # (N, d)

        # per-point scalar gate
        gate = torch.sigmoid((q * k).sum(dim=-1) * self.scale)  # (N,)
        out = v * gate.unsqueeze(-1)  # (N, d)
        out = self.out(out)           # (N, d)
        # residual + norm
        fused = self.norm(geom + out)
        return fused


def weighted_global_mean_pool(x, weight, batch, eps=1e-6):
    """
    x: (num_points, C)
    weight: (num_points,) non-negative
    batch: (num_points,)
    returns: (batch_size, C)
    """
    if weight.dim() == 2:
        weight = weight.squeeze(-1)
    num = scatter(x * weight.unsqueeze(-1), batch, dim=0, reduce='sum')
    den = scatter(weight, batch, dim=0, reduce='sum').unsqueeze(-1).clamp(min=eps)
    return num / den


class AuxFormer(nn.Module):
    """
    Full model that integrates geometry backbone with per-stage aux projections and point-wise cross-attention fusion.
    Input data shape: (B, N, C) where C >= 7 and the aux slice is at indices:
       aux = [zone, doppler, snr, local_density]  (so aux_dim=4)
    Configurable options:
      - apply_cross_attn: whether to fuse aux via cross-attn
      - use_snr_pooling: whether to use snr as weighting for final pooling
    """
    def __init__(self,
                 info,
                 dim_model=[32, 64, 128, 256, 512],
                 k=16,
                 aux_dim=4,
                 apply_cross_attn=True,
                 use_snr_pooling=True,
                 fps_ratio=0.25):
        """
        info: dict with 'num_classes' or 'num_keypoints'
        dim_model: list of dims for each stage (len = num_stages)
        aux_dim: number of aux channels per point
        """
        super().__init__()
        self.k = k
        self.aux_dim = aux_dim
        self.apply_cross_attn = apply_cross_attn
        self.use_snr_pooling = use_snr_pooling
        self.fps_ratio = fps_ratio

        in_channels = 3
        self.num_classes = info.get('num_classes', None)
        out_channels = self.num_classes if self.num_classes is not None else info.get('num_keypoints')

        # Input MLP
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)
        self.transformer_input = TransformerBlock(in_channels=dim_model[0], out_channels=dim_model[0])

        # Build downsampling + transformer stacks
        self.transition_down = nn.ModuleList()
        self.transformers_down = nn.ModuleList()
        for i in range(len(dim_model) - 1):
            self.transition_down.append(TransitionDown(in_channels=dim_model[i],
                                                       out_channels=dim_model[i + 1],
                                                       ratio=self.fps_ratio,
                                                       k=self.k))
            self.transformers_down.append(TransformerBlock(in_channels=dim_model[i + 1],
                                                           out_channels=dim_model[i + 1]))

        # Aux encoder: produce per-stage aux tokens that match geometry dims.
        if aux_dim > 0 and apply_cross_attn:
            stage_dims = [dim_model[0]] + [dim_model[i + 1] for i in range(len(dim_model) - 1)]
            self.aux_encoder = AuxEncoder(aux_dim=aux_dim, stage_dims=stage_dims)
            # create a PointCrossAttention module per stage
            self.cross_attn_blocks = nn.ModuleList([PointCrossAttention(d) for d in stage_dims])
        else:
            self.aux_encoder = None
            self.cross_attn_blocks = None

        # Output head
        if self.num_classes is None:
            # keypoint mode
            num_points = info['num_keypoints']
            self.num_points = num_points
            point_branches = {}
            for i in range(num_points):
                point_branches[f'branch_{i}'] = MLP([dim_model[-1], 64, 3], norm=None)
            self.mlp_output = nn.ModuleDict(point_branches)
        else:
            self.mlp_output = MLP([dim_model[-1], 64, out_channels], norm=None)

    def forward(self, data):
        """
        data: (B, N, C) where C >= 7 with order:
              [x,y,z, zone, doppler, snr, local_density]
        returns: logits (B, num_classes)
        """
        assert data.dim() == 3
        B, N, C = data.shape
        device = data.device

        # Extract coords and aux
        coords = data[:, :, :3]  # (B, N, 3)
        # aux ordering assumption: zone, doppler, snr, local_density
        if self.aux_dim > 0:
            aux = data[:, :, 3:3 + self.aux_dim]  # (B, N, aux_dim)
            aux_flat = aux.reshape(B * N, self.aux_dim)  # (B*N, aux_dim)
        else:
            aux = None
            aux_flat = None

        # Flatten coords to (B*N, 3) and build batch vector for PyG ops
        pos = coords.reshape(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)

        # Build initial features (dummy features)
        x = pos  # using coords as initial features is ok; you may want to pass zeros or learned features
        x_feat = self.mlp_input(x)  # (B*N, dim0)

        # initial graph & transformer
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x_feat = self.transformer_input(x_feat, pos, edge_index)

        # Prepare aux tokens per stage if available
        if self.aux_encoder is not None and aux_flat is not None:
            aux_stage_tokens = self.aux_encoder(aux_flat)  # list of (B*N, d_stage)
            # aux_stage_tokens[i] corresponds to geometry before or after stage i pooling (we'll downsample them)
        else:
            aux_stage_tokens = None

        # If cross-attn is used, we will maintain aux_stage_tokens downsampled alongside features.
        # Stage 0 fusion: before any TransitionDown
        if self.apply_cross_attn and aux_stage_tokens is not None:
            # stage 0 dims correspond to x_feat channel dim
            aux_tokens_stage0 = aux_stage_tokens[0]  # (B*N, d0)
            # ensure dims match by linear projection if needed (aux encoder already sized to match)
            fused = self.cross_attn_blocks[0](x_feat, aux_tokens_stage0)
            x_feat = fused

        # Backbone: loop through TransitionDown + Transformer
        curr_pos = pos
        curr_batch = batch
        curr_aux = aux_flat  # flattened aux aligned to curr_pos (for stage 0)
        curr_aux_tokens_list = aux_stage_tokens  # list of original per-point aux tokens (aligned to original pos)
        # For later stages we will compute aux_down for each stage using the TransitionDown pooling procedure
        for stage_idx, (td, tr) in enumerate(zip(self.transition_down, self.transformers_down)):
            # td will pool from current x_feat / curr_pos
            x_out, sub_pos, sub_batch, aux_out, id_clusters = td(x_feat, curr_pos, curr_batch,
                                                                aux=curr_aux_tokens_list[stage_idx] if curr_aux_tokens_list is not None else None)
            # x_out: (num_clusters_total, dim_next)
            # aux_out: (num_clusters_total, d_next) or None
            # Note: id_clusters are indices into curr_pos (so alignment preserved)
            # Update feature & pos & batch for next stage
            x_feat = x_out
            curr_pos = sub_pos
            curr_batch = sub_batch

            # local graph and transformer on downsampled points
            edge_index = knn_graph(curr_pos, k=self.k, batch=curr_batch)
            x_feat = tr(x_feat, curr_pos, edge_index)

            # cross-attn fusion for this stage if configured
            if self.apply_cross_attn and self.cross_attn_blocks is not None:
                # aux_out must be provided and match x_feat dim
                if aux_out is not None:
                    # aux_out shape: (num_clusters, d_stage) already matches stage dim (aux_encoder built heads accordingly)
                    x_feat = self.cross_attn_blocks[stage_idx + 1](x_feat, aux_out)
                else:
                    # No aux provided for this stage — skip fusion
                    pass

            # prepare curr_aux_tokens_list for next TD if needed:
            # We don't recompute aux_stage_tokens list here because aux_encoder produced original-level tokens.
            # But our TransitionDown returned aux_out per stage (pooled), so we don't need further mapping.

        # Final pooling. Optionally use SNR as weight. We need weights aligned to current x positions.
        if self.use_snr_pooling and (aux is not None):
            # aux originally was (B, N, aux_dim) -> flatten (B*N, aux_dim)
            # But after pooling we have curr_pos with num_final_points; we need to map SNR from original aux_flat to curr positions.
            # Our TransitionDown provided aux_out for the last stage; if present, extract SNR from that aux_out (if head included it)
            # Simpler: if the aux_out exists and contains SNR-like signal in same index (we didn't track SNR separately),
            # use the original SNR to map via 1-NN. For simplicity and robustness, re-map using nearest neighbor lookup.
            orig_pos = pos  # (B*N,3) original positions
            final_pos = curr_pos  # (M,3) downsampled
            # Build 1-NN mapping from final_pos -> orig_pos
            # Use knn_graph with k=1 but note signature expects batch info; we have batch vectors curr_batch and original batch
            # Build original batch vector
            orig_batch = batch
            # Build mapping edges: neighbors from orig_pos -> final_pos
            # We want for each final_pos the nearest orig index: knn_graph(x=orig_pos, y=final_pos, k=1,...)
            map_edges = knn_graph(x=orig_pos, y=final_pos, k=1, batch_x=orig_batch, batch_y=curr_batch)
            # map_edges: (2, E) with first row indices of final_pos, second row indices of orig_pos
            # Create mapped weights per final point:
            mapped_weights = torch.zeros(final_pos.shape[0], device=device)
            # original snr values
            snr_orig = aux_flat[:, 2] if aux_flat is not None else torch.ones(B * N, device=device)
            mapped_weights[map_edges[0]] = snr_orig[map_edges[1]].clamp(min=0.0)
            # normalize mapped_weights (optional)
            mapped_weights = torch.sigmoid(mapped_weights)  # bring to 0..1
            pooled = weighted_global_mean_pool(x_feat, mapped_weights, curr_batch)
        else:
            pooled = global_mean_pool(x_feat, curr_batch)

        # outputs
        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.mlp_output[f'branch_{i}'](pooled))
            y = torch.stack(y, dim=1)
            return y

        out = self.mlp_output(pooled)  # (B, num_classes)
        return out

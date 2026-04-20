# Implementation State — NeurIPS 2026 Revision
**Last updated: 2026-04-18**

This document captures the complete state of the codebase as of the NeurIPS 2026 revision effort. A fresh Claude instance reading this should be able to continue work without needing to retrace prior conversations.

---

## 1. Project Overview

**Goal:** Revise the ICML 2026 rejected paper "DGCNN-AFTNet" for NeurIPS 2026.

**Core reviewer criticisms addressed:**
- Zone feature was room-specific and unsubstantiated → removed entirely
- "Modality-agnostic" claim unproven → added IMU (chest-mounted, 6D) and BLE RSSI (3 beacons, 3D) as frame-level signals from a genuinely different sensor modality
- Weak baselines → added `dgcnn_aux_t` ablation and updated all aux baselines to 15D input
- Missing ablation → full 4-model ablation table now available

**Paper rename:** DGCNN-MMC-T (Multi-Modal Conditioning + Temporal)

---

## 2. Data Pipeline

### 2.1 Raw Data Location
```
/cluster/projects/kite/koorosh/Data/Koorosh-CareLab-Data/          ← original raw data (compute nodes only)
/cluster/projects/kite/koorosh/Data/Koorosh-CareLab-Data-Processed/ ← extracted per-subject pkl files
/cluster/projects/kite/koorosh/Data/Koorosh-CareLab-Data-Processed/stacked/ ← stacked/padded cache pkl files
```

### 2.2 Extractor: `misc/carelab_extractor_v2.py`
Replaces the old `carelab_extractor.py`. Key changes:
- **Zone removed** — was derived from XYZ using hand-defined room regions, not generalizable
- **IMU added** — loads `imu/acc.csv` and `imu/gyro.csv` per scenario, infers ms offsets by evenly distributing samples within each second
- **BLE added** — loads `imu/ble.csv` in long format (timestamp, mac_addr, rssi), pivots to wide using 3 known MAC addresses

**BLE MAC addresses (hardcoded):**
```python
BLE_BEACONS = ['AC:23:3F:AB:CA:2F', 'AC:23:3F:AB:CA:A4', 'AC:23:3F:F0:95:3A']
```

**Output per frame:**
```python
{
    "point_cloud":    np.array(N, 6),  # [X, Y, Z, Doppler, SNR, Density]
    "frame_signals":  np.array(9,),    # [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ble_1, ble_2, ble_3]
    "label":          str or -1,
    "timestamp":      int (epoch ms)
}
```

**Output files:** `{subject}_{scenario}.pkl` e.g. `1_1.pkl`, `3_7.pkl`
All 149 scenarios extracted with acc=OK, gyro=OK, ble=OK.

**SLURM script:** `extract.sh` (partition=gpu, 8G mem, 4 CPUs, 2h)
```bash
sbatch extract.sh
```

**Known issues fixed:**
- Subject 4 had NUL bytes (`\x00`) in CSVs → fixed with `line.replace('\x00', '')` generator wrapping csv.reader
- Some subjects had UTF-8 encoding errors → fixed with `encoding='utf-8', errors='replace'`

### 2.3 Dataset: `mmrnet/dataset/mmrnet_data.py`

**Only `MMRActionData` class was modified** (keypoint and identification classes untouched).

**Key changes from original:**
1. `raw_data_path` default → `/cluster/projects/kite/koorosh/Data/Koorosh-CareLab-Data-Processed`
2. All `"carelab"` checks use `.lower()` — path has capital C and L (`CareLab`)
3. Label key: `'y'` → `'label'`, point cloud key: `'x'` → `'point_cloud'`
4. Zone analysis block removed from `_process()`
5. `stack_and_padd_frames()` now stacks both `point_cloud (T, N, 6)` and `frame_signals (T, 9)` per sample, stores as `new_x` and `new_frame_signals`
6. `get()` returns `(point_cloud_tensor, frame_signals_tensor), y` — **x is a tuple**
7. `_select_points_by_density()` updated for 6-col format (density at col 5, no zone)
8. `_normalize_stack_by_centroid()` updated for 6-col format
9. `os.makedirs` added before saving stacked pkl to auto-create `stacked/` dir

**Critical: `get()` return format**
```python
# Dataset returns:
return (x, frame_signals), y
# where x.shape = (T, N, 6) or (T*N, 6) depending on use_temporal_format
# frame_signals.shape = (T, 9)

# Session wrapper sees:
x, y = batch[0], batch[1]
# batch[0] = (point_cloud_tensor, frame_signals_tensor)  ← tuple
# batch[1] = y
```

All models must unpack this tuple in their `forward` method.

### 2.4 Stacked Data Cache
Generated automatically on first training run. Filename encodes all parameters:
```
seed_{seed}_stacks_{T}_srate_{S}_maxpts_{N}_padd_{padding}_task_{task}.pkl
```
Current cache in use:
```
seed_20_stacks_40_srate_10_maxpts_22_padd_per_data_point_task_action.pkl
```

### 2.5 Frame Signal Analysis (diagnostic)
Run `misc/frame_signals_diagnostic.py` (requires compute node via `sbatch diagnose_signals.sh`).

**Key findings:**
- IMU acc std ≈ 0.1 — very low variation across actions (chest-mounted, hand activities)
- Gyro std ≈ 60 — raw sensor units (deg/s), 3 orders of magnitude larger than acc
- BLE std ≈ 10 dBm — encodes room position, not action
- MLP on frame_signals alone: **16.1% test accuracy** (chance = 3.3%) — signals have real discriminative content
- Per-modality: BLE alone 15.6%, Gyro alone 12.0%, ACC alone 5.1%
- **Scale mismatch is severe** → per-modality LayerNorm required (not single LayerNorm(9))

---

## 3. Architecture

### 3.1 Model Hierarchy

| Model | File | Description | Input |
|-------|------|-------------|-------|
| `dgcnn_aux` | `dgcnn_aux.py` | DGCNN + 15D concat, mean-pool over T | `(B,T,N,6)+(B,T,9)` → 15D flat |
| `dgcnn_aux_t` | `dgcnn_aux_t.py` | DGCNN + 15D concat + Temporal Transformer | `(B,T,N,6)+(B,T,9)` → 15D flat |
| `dgcnn_aux_fusion_t_v2` | `dgcnn_aux_fusion_t_v2.py` | DGCNN + point-level FiLM + Temporal Transformer | `(B,T,N,6)+(B,T,9)` (fs ignored) |
| `dgcnn_mmc_t` | `dgcnn_mmc_t.py` | DGCNN + point-level FiLM + frame-level cross-attn + Temporal Transformer | `(B,T,N,6)+(B,T,9)` |
| `p4transformer` | `p4transformer.py` | P4Transformer-style temporal baseline (pure PyTorch “P4DConv-lite” + Transformer) | XYZ-only: `(B,T,N,6)→xyz` |
| `p4transformer_aux` | `p4transformer.py` | Aux variant: uses all point aux + all frame aux | `(xyz + Doppler/SNR/Density + IMU/BLE)` = 15D |
| `sts_mixer` | `sts_mixer.py` | STS-Mixer–style: P4DConv-lite + spatial/temporal/channel mixer blocks | XYZ-only |
| `sts_mixer_aux` | `sts_mixer.py` | Aux variant: full point + frame aux | 15D |
| `ust_ssm` | `ust_ssm.py` | UST-SSM–style: P4DConv-lite + unified sequence (Mamba on `T×N` tokens; GRU fallback if unavailable) | XYZ-only |
| `ust_ssm_aux` | `ust_ssm.py` | Aux variant: full point + frame aux | 15D |

### 3.2 Proposed Model: `dgcnn_mmc_t` (DGCNN-MMC-T)

**Two-level conditioning hierarchy:**

**Point-level: FiLM (Feature-wise Linear Modulation)**
- Conditions each EdgeConv edge feature on per-point radar metadata: Doppler, SNR, Density
- `aux_dim=3` (no zone), applied inside `EdgeConvAuxLayer`
- Rationale: signals are co-located with each point, relationship is local and direct → FiLM is sufficient

**Frame-level: Cross-modal attention (`FrameCrossAttn`)**
- After global max-pool → per-frame embedding `E (B, T, D)`
- IMU/BLE are foreign-modality signals → content-dependent selection is needed → cross-attention
- Per-modality LayerNorm (acc, gyro, BLE each normalized independently to handle scale mismatch)
- Lightweight: `d_ca=64`, single-head, element-wise sigmoid gate
- Zero-initialized output projection (starts as identity)
- Rationale: "given what radar captured this frame, which aspects of subject's motion/position matter?"

**Ablation flags (independent):**
- `use_film_modulation` — disables point-level FiLM
- `use_frame_conditioning` — disables frame-level cross-attention
- `use_temporal_pos_embed` — disables learnable positional embeddings
- `temporal_layers=0` — disables temporal transformer
- `frame_modality_dims` — tuple of ints per modality for LayerNorm (default `(3,3,3)`)

### 3.3 `dgcnn_aux_fusion_t_v2`
Identical architecture to the original ICML submission `dgcnn_aux_fusion_t` except:
- Accepts `(point_cloud, frame_signals)` tuple input (`frame_signals` is ignored)
- `aux_dim=3` (no zone, only Doppler/SNR/Density)

### 3.4 `dgcnn_aux_t`
Same as `dgcnn_aux` but with temporal transformer instead of mean-pool over T.
- 15D concat input (broadcast frame_signals to all points)
- Used as ablation: isolates temporal transformer contribution without FiLM

### 3.5 Reviewer Defense for Architecture Choices

**Why FiLM at point level, not cross-attention?**
> "Point-level signals (Doppler, SNR, Density) vary per-point and are physically co-located with each radar reflection — the relationship is direct and local. FiLM is lightweight and sufficient for this granularity."

**Why cross-attention at frame level, not FiLM?**
> "IMU and BLE are foreign-modality signals from a different physical sensor. They describe the subject's global state, not any specific radar reflection. The content-dependence of attention is necessary — the radar representation queries the auxiliary signals to selectively retrieve relevant context. FiLM would impose a fixed scale+shift regardless of what the radar captured."

**Why not cross-attention at both levels?**
> "At the point level, cross-attention would lose the physical co-location inductive bias and be unnecessarily expressive for a signal that is directly co-located with each point."

---

## 4. Training Infrastructure

### 4.1 Runner: `runner.py`

Main SLURM job submission script. Key defaults:
```python
raw_data_path = '/cluster/projects/kite/koorosh/Data/Koorosh-CareLab-Data-Processed'
processed_data_base = '/cluster/projects/kite/koorosh/Data/Koorosh-CareLab-Data-Processed/stacked'
checkpoint_base = '/cluster/projects/kite/koorosh/Output/MiliPointCareLab/checkpoints'
use_temporal_format = True   # always on — all models handle flattening themselves
mail_type = 'FAIL'           # only email on failure
```

**Key flags:**
```bash
--model           # model name (see model_map in __init__.py)
--stacks          # T frames per sample
--sampling-rate   # stride between frames (10 = every 10th frame)
--cross-validation LOSO
--subject-id      # comma-separated for grid search e.g. 1,2,3,...,20
--mode            # train (default) or test
--begin           # defer start e.g. now+4hours, now+30minutes
--time            # wall time e.g. 0-6:0:0 (train), 0-0:30:0 (test)
```

**Experiment naming:** `{model}_{task}_seed{N}_stack{T}_srate{S}_LOSO_subj{ID}_opt{opt}_lr{lr}_bs{bs}_ep{ep}_wd{wd}`

**Processed data naming:** `seed_{N}_stacks_{T}_srate_{S}_maxpts_{N}_padd_{padding}_task_{task}.pkl`

**Checkpoint path fix (important):** `cli.py` line was `'checkpoints/' + a.save_name` (prepended local path to absolute path → mirror directory bug). Fixed to `a.save_name` directly.

**Test load path fix:** Same bug — `'checkpoints/' + a.load_name` → `a.load_name + '/'`.

### 4.2 Session Wrapper: `mmrnet/session/wrapper.py`

`test_step` was fixed to handle tuple input `x`:
```python
# batch_size must use x[0].shape[0] when x is a tuple
batch_size = x[0].shape[0] if isinstance(x, (tuple, list)) else x.shape[0]
# sample_input must handle tuple
self.sample_input = tuple(xi[0:1].detach().clone() for xi in x)  # if tuple
# top3_acc uses batch_size variable, not x.shape[0]
top3_acc = (top3 == y.unsqueeze(-1)).float().sum() / batch_size
```

### 4.3 Optimal Hyperparameters (found via subject 1 experiments)
```
stacks = 40
sampling_rate = 10    # stride — spans ~40 seconds, reduces temporal redundancy
learning_rate = 1e-4  # 1e-5 was too slow for frame conditioning pathway
batch_size = 128
max_epochs = 100      # overfits around epoch 40-50, checkpoint saves best
time = 0-6:0:0        # train, 0-0:30:0 test
```

---

## 5. Model Compatibility Status

All models must accept `(point_cloud, frame_signals), y` from the dataset.

### ✅ Updated (compatible with new pipeline)
| Model | Strategy | in_channels |
|-------|----------|-------------|
| `dgcnn_aux` | Broadcast fs → 15D, flatten T, mean-pool | 15 |
| `dgcnn_aux_t` | Broadcast fs → 15D, flatten T, temporal transformer | 15 |
| `dgcnn_aux_fusion_t_v2` | Unpack tuple, ignore fs, 6D point-level FiLM | 6 |
| `dgcnn_mmc_t` | Unpack tuple, 6D FiLM + 9D cross-attn | 6+9 |
| `p4transformer` | Temporal model, XYZ-only | 3 |
| `p4transformer_aux` | Temporal model, full aux (point+frame) | 15 |
| `sts_mixer` | Same contract as `p4transformer` | 3 |
| `sts_mixer_aux` | Same as `p4transformer_aux` | 15 |
| `ust_ssm` | Same contract as `p4transformer` | 3 |
| `ust_ssm_aux` | Same as `p4transformer_aux` | 15 |

### ✅ Updated 2026-04-15 (all now compatible)
| Model | File | Strategy |
|-------|------|----------|
| `pointnet_aux` | `pointnet_aux.py` | Broadcast fs → 15D, flatten `(B*T*N)`, mean-pool over T |
| `pointnext_aux` | `pointnext_aux.py` | Broadcast fs → 15D, flatten `(B*T*N)`, mean-pool over T |
| `deepgcn_aux` | `deepgcn_aux.py` | Broadcast fs → 15D, flatten `(B*T*N)`, mean-pool over T |
| `point_transformer_aux` | `point_transformer_aux.py` | Broadcast fs → 15D, flatten `(B*T*N)`, mean-pool over T |
| `pointmlp_aux` | `pointmlp_aux.py` | Special — flatten T into points: `(B, T*N, 15)` (no mean-pool, `self.points=T*N` unchanged) |
| `point_transformer_v3_aux` | `point_transformer_v3_aux.py` | Broadcast fs → 15D, flatten `(B*T*N)`, mean-pool over T |
| `pointmamba_aux` | `pointmamba_aux.py` | Broadcast fs → 15D, flatten `(B*T, N, 15)`, mean-pool over T |
| `mamba4d_aux` | `mamba4d_aux.py` | Broadcast fs → `(B, T, N, 15)`, pass directly (native temporal) |
| `mamba4d_aux_film` | `mamba4d_aux_film.py` | Same as mamba4d_aux; `aux_dim=3` (Doppler/SNR/Density cols 3:6) |

**Standard update pattern for flat models (Group A):**
```python
def forward(self, data):
    point_cloud, frame_signals = data       # unpack tuple
    B, T, N, _ = point_cloud.shape
    fs = frame_signals.unsqueeze(2).expand(-1, -1, N, -1)  # (B, T, N, 9)
    x = torch.cat([point_cloud, fs], dim=-1)               # (B, T, N, 15)
    x = x.reshape(B * T * N, self.in_channels)             # flatten
    batch = torch.arange(B * T, device=x.device).repeat_interleave(N)
    # ... rest of forward unchanged ...
    x = global_pool(x, batch)              # (B*T, D)
    x = x.view(B, T, -1).mean(dim=1)      # mean over T → (B, D)
    return self.output(x)
```
Also change `in_channels=7` → `in_channels=15` in `__init__`.

**Standard update pattern for temporal models (Group B — Mamba):**
```python
def forward(self, data):
    point_cloud, frame_signals = data       # unpack tuple
    B, T, N, _ = point_cloud.shape
    fs = frame_signals.unsqueeze(2).expand(-1, -1, N, -1)  # (B, T, N, 9)
    data = torch.cat([point_cloud, fs], dim=-1)             # (B, T, N, 15)
    # ... rest of forward unchanged, already handles (B, T, N, C) ...
```
Also change `in_channels=7` → `in_channels=15`.

---

## 6. Results

### 6.1 Ablation Table (20-subject LOSO mean ± std)

| Model | Temporal | Point FiLM | Frame CA | test_acc | merged_acc | f1_merged |
|-------|----------|-----------|---------|----------|------------|-----------|
| `dgcnn_aux` | ✗ | ✗ | ✗ | 0.691 ± 0.081 | 0.886 ± 0.057 | 0.859 ± 0.112 |
| `dgcnn_aux_t` | ✓ | ✗ | ✗ | 0.569 ± 0.085 | 0.818 ± 0.061 | 0.800 ± 0.103 |
| `dgcnn_aux_fusion_t_v2` | ✓ | ✓ | ✗ | **0.783 ± 0.035** | **0.929 ± 0.017** | **0.902 ± 0.088** |
| `dgcnn_mmc_t` | ✓ | ✓ | ✓ | 0.747 ± 0.042 | 0.911 ± 0.021 | 0.883 ± 0.096 |

### 6.2 Key Findings

1. **Temporal transformer alone hurts** (`dgcnn_aux` → `dgcnn_aux_t`: -12% test_acc). Without FiLM, the transformer models noisy 15D concat representations where frame signals are naively mixed with point features. The transformer amplifies noise rather than signal.

2. **Point-level FiLM is the single most impactful component** (`dgcnn_aux_t` → `dgcnn_aux_fusion_t_v2`: +21% test_acc). Conditioning each EdgeConv edge on Doppler/SNR/Density produces much richer per-frame representations for the temporal transformer to work with. Also dramatically reduces cross-subject variance (std drops from 0.085 to 0.035).

3. **Frame-level cross-attention shows neutral/slight negative effect** (`dgcnn_aux_fusion_t_v2` → `dgcnn_mmc_t`: -3.6% test_acc). IMU/BLE carry real discriminative signal (16% standalone accuracy vs 3.3% chance) but in LOSO the subject-specific IMU/BLE patterns slightly hurt generalization. The architecture is principled but the modality choice has limitations.

4. **Merged class accuracy is the better metric** — maps 30 actions into 4 clinical risk categories (L1-L4). This is more relevant for healthcare monitoring and shows higher numbers across all models.

### 6.3 Results Files
```
results/results_summary.csv          — all scalar metrics per model per subject
results/confusion_matrices.pkl       — dict[(model, subject_id)] → np.array(4,4)
```
Rebuild anytime: `python misc/collect_results.py`

### 6.4 Other Models (partial results — single subject only)
- `pointnet_aux` (subj 8): test_acc=0.804, merged_acc=0.942
- `pointnext_aux` (subj 8): test_acc=0.789, merged_acc=0.936
- `pointtransformer_aux` (subj 8): test_acc=0.719, merged_acc=0.915
- `pointmlp_aux` (subj 8): test_acc=0.708, merged_acc=0.892

---

## 7. Pending Work

### Immediate
- [x] Update remaining 9 aux models for new data format (done 2026-04-15)
- [ ] Run full LOSO for updated baselines

### Experiments
- [ ] Full ablation: disable FiLM only, disable frame CA only, disable both
- [ ] Per-feature ablation: remove Doppler, SNR, Density, IMU, BLE individually
- [ ] Frame conditioning mechanism comparison: FiLM vs cross-attn vs none at frame level

### Paper
- [ ] Rewrite methodology section with two-level conditioning hierarchy
- [ ] Add limitations and impact statement
- [ ] Aggregate confusion matrix across all 20 LOSO folds
- [ ] Per-class F1 scores (30 classes)
- [ ] Fix typos: "molality" → "modality", "axillary" → "auxiliary"
- [ ] Unify k-NN to k=20 (N_max=22, k=30 is impossible)

---

## 8. File Map

```
MiliPoint-Caregiver/
├── mm.py                                    # CLI entry point
├── runner.py                                # SLURM job submission
├── mmrnet/
│   ├── cli.py                               # argument parsing + dispatch
│   ├── dataset/
│   │   └── mmrnet_data.py                   # MMRActionData (only class modified)
│   ├── models/
│   │   ├── __init__.py                      # model registry
│   │   ├── dgcnn_aux.py                     # ✅ 15D baseline, no temporal
│   │   ├── dgcnn_aux_t.py                   # ✅ 15D + temporal transformer
│   │   ├── dgcnn_aux_fusion_t.py            # ❌ OLD (7D + zone, ICML version, keep for reference)
│   │   ├── dgcnn_aux_fusion_t_v2.py         # ✅ 6D FiLM + temporal (no zone, no frame CA)
│   │   └── dgcnn_mmc_t.py                   # ✅ proposed model: FiLM + cross-attn + temporal
│   └── session/
│       ├── wrapper.py                       # ModelWrapper — test_step fixed for tuple input
│       └── train.py                         # training loop
├── misc/
│   ├── carelab_extractor_v2.py              # data extraction with IMU/BLE
│   ├── collect_results.py                   # parse test outputs → CSV + confusion matrix pkl
│   ├── frame_signals_diagnostic.py          # MLP test: can frame_signals predict actions?
│   └── inspect_stacked_data.py              # analyze stacked pkl statistics
├── configs/action/                          # TOML configs (use runner.py defaults instead)
├── results/
│   ├── results_summary.csv                  # current results
│   └── confusion_matrices.pkl               # confusion matrices keyed by (model, subject_id)
├── docs/
│   ├── neurips2026_revision_plan.md         # full revision plan
│   └── implementation_state.md             # this file
├── extract.sh                               # SLURM script for carelab_extractor_v2
├── diagnose_signals.sh                      # SLURM script for frame_signals_diagnostic
└── inspect_stacked.sh                       # SLURM script for inspect_stacked_data
```

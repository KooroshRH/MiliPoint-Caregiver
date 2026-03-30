# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiliPoint-Caregiver is a research package for healthcare action recognition using mmWave radar point clouds and Graph Neural Networks. It processes point cloud data from millimeter-wave radar for privacy-preserving continuous monitoring (30 action classes, 17 keypoint estimation, person re-identification).

## Commands

### Training
```bash
python mm.py train mmr_act dgcnn_aux_fusion_t \
    --dataset_config configs/action/mmr_action_stack_40_point_carelab_5fold_0.toml \
    --batch-size 32 --learning-rate 1e-4 --max-epochs 100 --accelerator gpu
```

### Testing
```bash
python mm.py test mmr_act dgcnn_aux_fusion_t \
    --load lightning_logs/version_XXX/checkpoints/best.ckpt \
    --dataset_config configs/action/mmr_action_stack_40_point_carelab_5fold_0.toml \
    --accelerator gpu
```

### SLURM Job Submission
```bash
python runner.py --model dgcnn_aux_fusion_t --task mmr_act --stacks 40 \
    --batch-size 32 --learning-rate 1e-4 --max-epochs 100 --submit
```

### CLI Syntax
```
python mm.py <action> <dataset> <model> [options]
# action: train | test | eval
# dataset: mmr_act | mmr_kp | mmr_iden
# model: see model registry below
```

### Package Installation
```bash
pip install -e .
```

## Architecture

### Data Flow
Point clouds have 7 channels: `[X, Y, Z, Zone, Doppler, SNR, Density]`
- Channels 0-2: geometric coordinates
- Channels 3-6: auxiliary radar metadata

Temporal format stored as `(T, N, C)` — T frames (5–60), N=22 points, C=7 channels. Flattened to `(T*N, C)` for single-frame models.

### Key Modules

- **`mmrnet/cli.py`**: All argument parsing and dispatch logic
- **`mmrnet/models/`**: 31 model architectures; `__init__.py` is the model registry
- **`mmrnet/dataset/mmrnet_data.py`**: Dataset classes (`MMRActionData`, `MMRKeypointData`, `MMRIdentificationData`) and preprocessing
- **`mmrnet/session/wrapper.py`**: PyTorch Lightning `ModelWrapper` — handles loss, metrics (accuracy, MLE), checkpointing, and confusion matrices
- **`mmrnet/session/explainability.py`**: Model interpretability and visualization
- **`runner.py`**: SLURM grid search submission with hyperparameter sweeps

### Proposed Model: DGCNN_AFTNet (`dgcnn_aux_fusion_t`)
Located in `mmrnet/models/dgcnn_aux_fusion_t.py`. Three key innovations:
1. Auxiliary-Guided FiLM Modulation (Zone/Doppler/SNR/Density → scale/shift parameters)
2. Temporal Transformer Encoder across T frames
3. Learnable Temporal Positional Embeddings

### Configuration System
TOML configs in `configs/` control dataset paths, cross-validation strategy, and temporal stacking. Naming convention: `mmr_action_stack_{T}_point_carelab_{cv}_{fold}.toml`

Cross-validation modes:
- `5-fold`: folds 0–4 available
- `LOSO`: leave-one-subject-out
- `holdout`: standard 80-10-10 split

## Ablation Flags (for `dgcnn_aux_fusion_t`)

| Flag | Effect |
|------|--------|
| `--model-temporal-layers 0` | Disable temporal transformer |
| `--model-no-film-modulation` | Disable FiLM modulation |
| `--model-no-temporal-pos-embed` | Disable learnable positional embeddings |
| `--model-aux-dim 0` | Use only XYZ, no auxiliary features |
| `--model-k 10,20,30` | Grid search over k-NN parameter |

## Training Infrastructure

- Framework: PyTorch Lightning 1.9.3
- Logs: `lightning_logs/version_*/` (1200+ experiments)
- Checkpoints: `checkpoints/` or `lightning_logs/version_*/checkpoints/best.ckpt`
- Outputs: `outputs/` (50+ experiment result directories)
- Experiment tracking: Weights & Biases (`wandb`)

## Available Models

Baselines: `dgcnn`, `pointnet`, `pointtransformer`, `mlp`
Auxiliary variants: `dgcnn_aux`, `pointnet_aux`, `pointtransformer_aux`, `pointmlp_aux`, `pointnext_aux`, `deepgcn_aux`, `pointmamba_aux`, `point_transformer_v3_aux`, `mamba4d_aux`, `mamba4d_aux_film`
Proposed: `dgcnn_aux_fusion_t`, `dgcnn_aux_fusion_stattn`
Other: `ensemble_model`, `attdgcnn`, `hybrid_pointnet`, `aux_former`, `pointnet_film`

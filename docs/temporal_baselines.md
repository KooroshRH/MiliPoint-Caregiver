# Temporal Baseline Candidates (External Repos) — Integration Notes

**Last updated:** 2026-04-16  
**Context:** Our dataset returns `((point_cloud, frame_signals), y)` where:
- `point_cloud`: `(B, T, N, 6)` with channels `[X, Y, Z, Doppler, SNR, Density]`
- `frame_signals`: `(B, T, 9)` with channels `[acc(3), gyro(3), BLE(3)]`

These baselines are *temporal by nature* (designed for point cloud sequences / videos). They are not “flatten T and treat as one cloud” baselines like our non-temporal aux models.

This document captures:
1. **Idea** (what the method contributes)
2. **How to clone/build** (high-level)
3. **How to integrate** into our codebase and make it compatible with our data format

---

## Common integration strategy in our repo

### Registry / naming (avoid runner failures)
- The only valid `--model` names are the keys in `mmrnet/models/__init__.py:model_map`.
- Before submitting jobs for a newly-added baseline, list keys with:
  - `python misc/list_registered_models.py`

### Where the model must land
- Add an adapter model in `mmrnet/models/`, e.g. `mmrnet/models/p4transformer_adapter.py`
- Register it in `mmrnet/models/__init__.py` `model_map` under a clear key
- Ensure the adapter `forward(self, data)` accepts the tuple:
  - `point_cloud, frame_signals = data`

### “Vendor vs port vs re-implement” decision rule
- **Vendor (submodule/subtree)**: best when the repo has custom CUDA ops and a working training-time forward path we can reuse.
- **Port (copy a small set of modules into `mmrnet/models/`)**: best when the model is pure PyTorch and self-contained.
- **Re-implement**: best when the repo depends on TF1, or when the “idea” is clearer than the code, or the codebase is too invasive.

### What to feed these temporal models
Most external temporal baselines expect **XYZ-only** sequences:
- Use `xyz = point_cloud[..., :3]` → shape `(B, T, N, 3)`

Optional: if the baseline supports per-point features, we can also supply
- `point_feats = point_cloud[..., 3:]` → `(B, T, N, 3)` (Doppler/SNR/Density)

For frame signals (`frame_signals`), default plan is:
- **ignore initially** for strict baseline parity (these papers do not use IMU/BLE)
- later add a controlled variant if we want (e.g., concatenate after frame pooling)

### Adapter skeleton (what we’ll implement repeatedly)
Most temporal baselines can be made compatible with a thin wrapper like:

```python
def forward(self, data):
    point_cloud, frame_signals = data
    xyz = point_cloud[..., :3]      # (B, T, N, 3)
    # Option A (most repos): pass xyz directly
    logits = self.backbone(xyz)
    return logits
```

If the external model expects flattened `(B, T*N, 3)` or different layout, the adapter is responsible for reshaping/permuting only (no behavior changes).

### Adapter contract (keep ports uniform)
- **Input**: `data == (point_cloud, frame_signals)` where `point_cloud` is `(B,T,N,6)` and `frame_signals` is `(B,T,9)`
- **Default baseline**: XYZ-only (`xyz = point_cloud[..., :3]`); ignore `frame_signals`
- **Output**: logits `(B, num_classes)` (30 for our mmr_act task)
- **Temporal reduction**: only if the external backbone outputs per-frame logits/features; use mean over \(T\) unless the method defines its own temporal head
- **No hidden behavior changes**: adapters reshape/permutation + head replacement only; all architectural choices stay in the vendored/ported code

### Data/time conventions
External repos often assume:
- fixed `T` (e.g. 32 frames) and larger `N` (e.g. 256/1024 points)
Our case:
- `T=40`, `N=22` (very sparse). Some methods may need hyperparameter adjustment (patch size, neighborhood radius, KNN k, etc.).

### Reuse observation (important)
Several “temporal baselines” here share the **same core building block**: `P4DConv` / “point_4d_convolution”.
- `P4Transformer`, `PST-Transformer`, `STS-Mixer`, and even `UST-SSM` all import `point_4d_convolution`.
- This means once we get one `P4DConv` pipeline building/running cleanly inside our repo, the others become much easier to port.

---

## 1) P4Transformer (CVPR 2021)

- **Repo**: `https://github.com/hehefan/P4Transformer`  
- **README**: `https://raw.githubusercontent.com/hehefan/P4Transformer/main/README.md`  

### Idea
P4Transformer avoids explicit tracking by:
- **Point 4D convolution** to embed local spatio-temporal neighborhoods in raw point cloud videos
- A **Transformer** over embedded local tokens to capture global motion/appearance across the clip

### Clone/build (high level)
The repo uses custom CUDA ops (PointNet++-style FPS + radius search). From README:
- build “modules” via `python setup.py install`

### Code touchpoints (what to read/port first)
- **Model**: `models/msr.py` defines `class P4Transformer(nn.Module)` with:
  - `forward(self, input):  # input is [B, L, N, 3]`
  - uses `P4DConv` then adds a **(x,y,z,t)** positional embedding and a Transformer
- **Train entry**: `train-msr.py` (MSRAction3D)
- **CUDA ops**: `modules-pytorch-*/` and `modules/point_4d_convolution.py` (imported as `from point_4d_convolution import *`)

### Compatibility plan (our data)
1. **Adapter input**:
   - `xyz = point_cloud[..., :3]` (B,T,N,3)
2. **Tokenization / neighborhood ops**:
   - P4Transformer’s 4D conv expects meaningful neighborhoods; with `N=22`, radius/FPS settings likely must be reduced.
3. **Minimal integration approach**:
   - Vendor the model code into `external/P4Transformer/` (git submodule or subtree), then wrap it with an adapter class in `mmrnet/models/`.
4. **Output head**:
   - Replace/override the classification head to `num_classes = info['num_classes']` (30 actions in our task).
5. **Exact shape match**:
   - P4Transformer expects `input.shape == (B, L, N, 3)` where L is frames.
   - Our `xyz` already matches.

### Risks / gotchas
- Heavy reliance on PointNet++ CUDA kernels; cluster build toolchain must be aligned.
- Hyperparameters from MSRAction3D (hundreds of points) won’t directly transfer to `N=22`.

### Vendor plan (recommended)
- **Approach**: git submodule under `external/P4Transformer/` pinned to a commit
- **Build**: one-time compile of CUDA ops during environment setup (document exact CUDA/PyTorch combo)
- **Integration surface**: adapter file + minimal head override; avoid editing vendored code unless required for import paths

---

## 2) PST-Transformer

- **Repo**: `https://github.com/hehefan/PST-Transformer`  
- **README**: `https://raw.githubusercontent.com/hehefan/PST-Transformer/main/README.md`  

### Idea
PST-Transformer models point cloud videos by:
- global self-attention across points while **preserving spatio-temporal structure**
- **decoupled spatio-temporal encoding** (timestamps provide order, spatial coords are unordered)
- designed to search related points across the entire clip (no tracking)

### Clone/build (high level)
Like P4Transformer, it uses PointNet++-style CUDA operators:
- compile under `modules/` with `python setup.py install`

### Code touchpoints (what to read/port first)
- **Model**: `models/sequence_classification.py` defines `class PSTTransformer(nn.Module)`:
  - `forward(self, input):  # input is [B, L, N, 3]`
  - uses `P4DConv`, then a Transformer that consumes `(xyzs, features)`
- **Train entry**: `train-msr-small.py`, `train-ntu60.py`
- **CUDA ops**: `modules/point_4d_convolution.py` + related kernels

### Compatibility plan (our data)
1. Start XYZ-only: `xyz = point_cloud[..., :3]` (B,T,N,3)
2. Align expected input format:
   - Many PST-style implementations flatten to `(B, T*N, 3)` plus time indices; we can derive time indices trivially from `T`.
3. Register as e.g. `pst_transformer` in `model_map`.
4. Practical note:
   - The official PST-Transformer repo already accepts `[B, L, N, 3]`, so we likely do **not** need to flatten.

### Risks / gotchas
- Same sparse-point issue (`N=22`) likely needs smaller neighborhoods / fewer tokens.
- CUDA ops build requirements.

### Vendor plan (recommended)
- **Approach**: vendor alongside P4Transformer, reusing the same CUDA op toolchain where possible (shared `P4DConv` concept)
- **Integration surface**: adapter + registry entry; keep external training scripts unmodified

---

## 3) STS-Mixer (CVPR 2026 Findings)

- **Repo**: `https://github.com/Vegetebird/STS-Mixer`  
- **README**: `https://raw.githubusercontent.com/Vegetebird/STS-Mixer/main/README.md`  
- **Paper**: linked in README (`https://arxiv.org/pdf/2604.11637`)

### Idea
STS-Mixer proposes a **mixer-style** architecture for 4D point cloud videos, mixing along:
- **spatial**
- **temporal**
- **spectral / channel**

It builds on PST-Transformer / P4Transformer per README.

### Clone/build (high level)
From README, notable dependencies:
- PyTorch 2.1.1 + CUDA 11.8 wheels
- PyTorch3D
- PointNet2 ops (erikwijmans/Pointnet2_PyTorch)
- a custom module under `model/module` with `python setup.py install`

### Code touchpoints (what to read/port first)
- **Entry**: `main.py`
- **Model**: `model/stsmixer.py`
  - defines `class Model(nn.Module)`
  - `forward(self, x)` expects `x` shaped `(B, F, N, C)` where `C=3` for xyz
  - uses `P4DConv` from `model/module/point_4d_convolution.py`
  - uses **PyTorch3D KNN** and a graph Fourier transform (GFT) step (spectral mixing)

### Compatibility plan (our data)
1. XYZ-only baseline first: `xyz = point_cloud[..., :3]`
2. Identify STS-Mixer’s expected tensor shape (likely `(B, T, N, 3)` or `(B, T*N, 3)`).
3. Implement an adapter model class that:
   - prepares the expected input structure
   - uses our `info['num_classes']`
4. Add a runner-compatible training entry by registering in `model_map`.
5. Heads-up:
   - STS-Mixer’s default `num_classes` in code is 20 (MSRAction3D); we must set it to 30.

### Risks / gotchas
- More dependencies than PST/P4 (PyTorch3D + custom module build).
- Sparse `N=22` may need re-tuning patch/token sizes.

### Vendor plan (recommended)
- **Approach**: vendor repo (submodule/subtree) because it has multiple compiled deps
- **Integration surface**: adapter that feeds `(B,T,N,3)`; set `num_classes=30`
- **Pinned deps**: record exact versions (PyTorch/CUDA/PyTorch3D) used to build successfully

---

## 4) PST2

- **Status**: no implementation found yet (as of 2026-04-16)

### Plan
- Track down an official/unofficial repo, or implement from paper if necessary.
- Once found, treat similarly: vendor repo → adapter → register.

---

## 5) UST-SSM (ICCV 2025)

- **Repo**: `https://github.com/wangzy01/UST-SSM`  
- **README**: `https://raw.githubusercontent.com/wangzy01/UST-SSM/main/README.md`

### Idea
UST-SSM adapts modern **State Space Models (SSMs)** (Mamba-like) to point cloud videos by:
- reorganizing points into semantic-aware sequences (**STSS** scanning)
- aggregating spatio-temporal structure (**STSA**)
- enhancing temporal interaction (**TIS**)

### Clone/build (high level)
From README:
- `pip install causal-conv1d mamba-ssm`
- compile custom CUDA ops:
  - PointNet++ layers under `modules/`
  - KNN_CUDA wheel

### Code touchpoints (what to read/port first)
- **Train entries**: `msr-train.py`, `ntu-train.py`
- **Model**: `models/UST.py` defines `class UST(nn.Module)`
  - also uses **`P4DConv` tube embedding**
  - then constructs a **prompt-guided routing** over tokens and runs Mamba/SSM blocks
  - expects the input like the other P4DConv-based repos: `(B, L, N, 3)`

### Compatibility plan (our data)
1. Provide `xyz = point_cloud[..., :3]` (B,T,N,3)
2. If the model expects per-point features, optionally pass Doppler/SNR/Density as extra channels (second phase).
3. Adapter responsibilities:
   - map our shape/ordering to their scanning pipeline
   - set classification head to our `num_classes`
4. Pragmatic porting plan:
   - first get UST to run with *their* default hyperparams on our shape, then tune token counts / grouping for `N=22`.

### Risks / gotchas
- Heavier dependency surface (mamba + CUDA ops + KNN_CUDA).
- Likely tuned for larger `N`; may require changing prompt-guided clustering / token count assumptions.

### Vendor plan (recommended)
- **Approach**: vendor repo, but try to share `P4DConv`/PointNet++ op builds with PST/P4 if feasible
- **Integration surface**: adapter + head override; keep Mamba/SSM blocks untouched

---

## 6) 3DInAction (CVPR 2024)

- **Repo**: `https://github.com/sitzikbs/3dincaction`  
- **README**: `https://raw.githubusercontent.com/sitzikbs/3dincaction/main/README.md`

### Idea
3DInAction is an action-recognition framework in 3D point clouds that supports:
- per-frame and per-clip classification
It is not necessarily “point-cloud-video-native” in the same way as PST/P4, but it provides temporal clip modeling pipelines and training code.

### Clone/build (high level)
From README:
- PyTorch 1.10.1, CUDA 11.3
- compile PointNet2 FPS modules via `python setup.py install` in `./models`
- uses `wandb` by default

### Compatibility plan (our data)
Two integration options:
1. **Model-only port**: copy the relevant network definition into `mmrnet/models/` and use our trainer (preferred).
2. **Training pipeline port**: less preferred, because it conflicts with our Lightning-based `mm.py` ecosystem.

Adapter:
- use `xyz = point_cloud[..., :3]` and keep `T` as temporal dimension
- ignore `frame_signals` initially

### Code touchpoints (what to read/port first)
- `models/P4Transformer.py`, `models/PST_Transformer.py` (they vendor their own P4/PST modules)
- `models/setup.py` (builds a native extension for FPS/ops)
- `train.py` / `test.py` (their standalone training pipeline — we likely won’t use it)

### Note
Their `models/PST_Transformer.py` expects input shaped `(B, T, 3, N)` then permutes to `(B, T, N, 3)`.
Our adapter can supply `(B, T, N, 3)` directly and bypass that layout.

### Vendor plan (recommended)
- **Approach**: **port model-only** into `mmrnet/models/` if the dependency surface is small; otherwise vendor repo
- **Integration surface**: keep our Lightning training loop; do not port their training pipeline

---

## 7) Kinet (CVPR 2022)

- **Repo**: `https://github.com/jx-zhong-for-academic-purpose/Kinet`  
- **README**: `https://raw.githubusercontent.com/jx-zhong-for-academic-purpose/Kinet/main/README.md`

### Idea
Kinet classifies point cloud sequences using *static* backbones by fitting **feature-level space-time (ST) surfaces**:
- it “unrolls” a solver for ST-surfaces in feature space
- couples a static branch + dynamic branch (+ fusion)

### Clone/build (high level)
Important constraint from README:
- Training is based on **TensorFlow 1.x** with custom TF ops
- PyTorch CPU only used for reading data in that repo

### Code touchpoints (what to read/port first)
- `shrec2017_release/model_cls_static.py` and `model_cls_dynamic.py`
  - Both use TF placeholders `point_cloud: [B, num_point * num_frames, 3]`
  - The dynamic model builds a temporal module via custom TF ops under `tf_ops/`

### Compatibility plan (our data)
Because our project is PyTorch/Lightning:
- **Preferred**: re-implement the Kinet idea in PyTorch around our existing static backbones (PointNet/PointNeXt/etc.).
  - This avoids TF1 and TF custom ops entirely.
- If we still want to vendor code:
  - isolate it under `external/Kinet/` and treat it as a separate experiment pipeline (not ideal for NeurIPS revision timing).

If re-implementing:
1. Static backbone produces per-frame features: `f_t = backbone(xyz_t)`
2. Dynamic module fits feature-level ST surfaces across time
3. Classification head + optional fusion

### Practical “compat” mapping (if we ever run their TF pipeline)
- Convert our `(B, T, N, 3)` into `point_cloud_flat = (B, T*N, 3)` to match their placeholder shape.
- But we would still need TF1 + custom ops, so this is mainly for reference.

### Vendor plan (recommended)
- **Approach**: **re-implement in PyTorch** (treat TF1 code as a reference only)
- **Integration surface**: add a `kinet_like_*` model in `mmrnet/models/` that wraps an existing static backbone + dynamic ST-surface module

---

## Practical next steps (recommended order)

1. **P4Transformer / PST-Transformer / STS-Mixer** (closest to our domain; PyTorch; widely used)
2. **UST-SSM** (strong temporal baseline but heavier deps)
3. **3DInAction** (if we can cleanly port just a model variant)
4. **Kinet** (likely needs PyTorch re-implementation to be practical)

---

## Notes on fairness / reporting

For NeurIPS revision comparisons:
- Baseline versions should start with **XYZ-only** input to match the external papers.
- If we create “+aux” variants using Doppler/SNR/Density or IMU/BLE, report them as *our* extensions (not as the baseline).


# NeurIPS 2026 Revision Plan: DGCNN-AFTNet v2

Based on ICML 2026 reviews (Submission #13311, scores: 2/3/3/2). Target: NeurIPS 2026.

---

## 1. Review Diagnosis

### Critical issues (raised by all 4 reviewers)
| Issue | Reviewers | Status |
|-------|-----------|--------|
| "Modality-agnostic" claim unsubstantiated — only radar-derived aux tested | QTbR, MYDw, Xiwx, fJX4 | **Fix: add IMU + BLE modalities** |
| Zone feature encodes room-specific priors, layout-invariance unproven | QTbR, MYDw, Xiwx, fJX4 | **Fix: remove zone, drop claim, discuss in limitations** |
| k=20 vs k=30 inconsistency (k=30 impossible with N_max=22) | QTbR, MYDw, Xiwx, fJX4 | **Fix: clarify and unify** |
| Missing limitations section / impact statement | QTbR, MYDw, Xiwx, fJX4 | **Fix: write it** |

### Major issues (raised by 2-3 reviewers)
| Issue | Reviewers | Status |
|-------|-----------|--------|
| No per-feature ablation (Doppler, SNR, Density, Zone individually) | MYDw, Xiwx | **Fix: full per-feature ablation matrix** |
| Weak temporal baseline comparison — need "DGCNN + same Transformer" control | MYDw, Xiwx, fJX4 | **Fix: add matched temporal baselines** |
| Missing PST2 / P4Transformer comparisons | fJX4 | **Fix: add or justify exclusion** |
| Per-class/per-fold diagnostics missing (confusion matrix only for Subject 8) | MYDw, Xiwx | **Fix: aggregate confusion matrix, per-class F1** |
| Reproducibility: preprocessing/padding for N_max=22, train/val split unclear | MYDw | **Fix: clarify in paper** |
| Typos: "molality-agnostic", "axillary", notation inconsistencies | QTbR, MYDw | **Fix: proofread** |

---

## 2. Data Changes

### Remove: Zone ID
- Zone was derived from XYZ coordinates using hand-defined room regions
- Cannot support layout-invariance claim with single-room data
- Drop from auxiliary vector and from all generalization claims
- Discuss as limitation: "future work could explore learned spatial context"

### Add: IMU (6D) — frame-level
- Source: chest-mounted device on each subject
- Channels: accelerometer (3) + gyroscope (3) = 6 values per frame
- Alignment: match IMU timestamps to radar frame timestamps in `carelab_extractor.py`
- Nature: **frame-level** signal (one reading per temporal frame, same for all N points)

### Add: BLE RSSI (3D) — frame-level
- Source: 3 BLE beacons around the room
- Channels: 3 RSSI values (one per beacon) per frame
- Alignment: match BLE timestamps to radar frame timestamps in `carelab_extractor.py`
- Nature: **frame-level** signal (subject-level, not point-level)

### Revised input structure
```
Point-level (per point, per frame):
  Geometric: XYZ                    (3 channels)
  Auxiliary:  Doppler, SNR, Density  (3 channels)

Frame-level (per frame, shared across all points):
  IMU: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z  (6 channels)
  BLE: rssi_beacon_1, rssi_beacon_2, rssi_beacon_3         (3 channels)
```

---

## 3. Architecture Changes

### 3.1 Point-level conditioning: FiLM (unchanged from v1)

Retain existing FiLM mechanism in EdgeConv layers. For each edge (i,j):
```
e_aux = [aux_i, aux_j]            # aux = [Doppler, SNR, Density]
[gamma, beta] = MLP_aux(e_aux)
h'_ij = gamma * MLP_geom(e_geom) + beta
```

**Rationale:** Point-level signals (Doppler, SNR, Density) vary per-point and naturally condition edge-level geometric reasoning. FiLM is lightweight and proven effective for this granularity.

### 3.2 Frame-level conditioning: AttnRes-style cross-modal attention (NEW)

After global max-pooling produces frame embeddings `E_t ∈ R^D`, condition them on IMU+BLE using depth-selective attention inspired by Attention Residuals (Kimi Team, 2026).

**Design:**

Process frame-level auxiliary signals through a small MLP stack to produce multi-level representations:
```
a0 = [IMU_t, BLE_t]   ∈ R^9       # raw signals
a1 = MLP_1(a0)        ∈ R^D_aux   # first-level features
a2 = MLP_2(a1)        ∈ R^D_aux   # second-level features
```

Apply AttnRes-style selective aggregation — the frame embedding generates a query that attends over auxiliary representations at different depth levels:
```
q_t = W_q · E_t                                           # (D_aux,)
K = RMSNorm(stack([a0', a1, a2]))                         # (3, D_aux)  [a0' = Linear(a0) to project to D_aux]
alpha = softmax(q_t^T · K / sqrt(D_aux))                  # (3,)
aux_context = sum(alpha_i * V_i)                           # (D_aux,)
E'_t = E_t + W_o(aux_context)                             # residual connection
```

**Rationale:** Inspired by AttnRes's insight that fixed-weight aggregation (analogous to simple concatenation or FiLM) uniformly mixes all representation levels, while learned attention allows selective retrieval. The spatial representation decides at each frame whether it needs raw sensor readings (a0), low-level processed features (a1), or higher-level abstractions (a2). This is the depth-wise analog applied to cross-modal conditioning.

**Why not FiLM for frame-level?** FiLM produces a single scale+shift — appropriate for point-level where the conditioning signal is local and varies per-edge. Frame-level signals are global context that should be selectively integrated based on what the spatial representation already captured. Attention-based selection is more expressive for this purpose.

**Why not simple cross-attention?** Simple cross-attention over a single auxiliary representation is a special case (depth=1). AttnRes-style multi-level attention over the MLP stack gives the model access to different abstraction levels of the auxiliary signal, and the learned query naturally selects the right level.

### 3.3 Full revised forward pass

```
INPUT: (B, T, N, C_total)
  Split into:
    geom:      (B, T, N, 3)   — XYZ
    point_aux: (B, T, N, 3)   — Doppler, SNR, Density
    frame_aux: (B, T, 9)      — IMU(6) + BLE(3)

SPATIAL BACKBONE (per frame, flattened to B*T frames):
  EdgeConv_1(geom, point_aux) + FiLM → x1   (B*T*N, 32)
  EdgeConv_2(x1, point_aux)   + FiLM → x2   (B*T*N, 32)
  EdgeConv_3(x2, point_aux)   + FiLM → x3   (B*T*N, 32)
  Concat [x1, x2, x3] → Linear → x_dense    (B*T*N, 1024)
  Global max pool per frame → E              (B, T, 1024)

FRAME-LEVEL CONDITIONING (AttnRes-style):
  frame_aux → MLP stack → [a0', a1, a2]     (B, T, 3, D_aux)
  q = W_q(E)                                 (B, T, D_aux)
  alpha = softmax(q @ RMSNorm([a0', a1, a2])) (B, T, 3)
  aux_ctx = weighted sum                      (B, T, D_aux)
  E' = E + W_o(aux_ctx)                      (B, T, 1024)

TEMPORAL MODELING:
  E' + learnable positional embeddings        (B, T, 1024)
  Temporal Transformer Encoder (L=1, H=4)     (B, T, 1024)
  Mean pool over T                            (B, 1024)

CLASSIFICATION HEAD:
  MLP [1024, 1024, 256, 128, 30]             (B, 30)
```

---

## 4. Baseline Comparison Strategy

### Principle: all models receive the same information

For baselines, broadcast frame-level signals (IMU, BLE) to every point in the frame and concatenate:
```
Baseline input per point: [x, y, z, doppler, snr, density, imu(6), ble(3)] = 15D
```

For DGCNN-AFTNet v2: point-level (6D) + frame-level (9D) processed separately as described above.

### Required baselines

| Model | Input | Temporal | Purpose |
|-------|-------|----------|---------|
| PointNet++ (15D concat) | 15D | none (pool) | MLP-based baseline |
| PointMLP (15D concat) | 15D | none (pool) | MLP-based baseline |
| PointNeXt (15D concat) | 15D | none (pool) | Modern MLP baseline |
| DGCNN (15D concat) | 15D | none (pool) | Graph-based baseline (geometric foundation) |
| DeepGCN (15D concat) | 15D | none (pool) | Deep graph baseline |
| Point Transformer (15D concat) | 15D | none (pool) | Attention-based baseline |
| PTv3 (15D concat) | 15D | none (pool) | SOTA attention baseline |
| Mamba4D (15D concat) | 15D | SSM temporal | Temporal SSM baseline |
| **DGCNN + Temporal Transformer (3D only)** | 3D | Transformer | **NEW: isolates temporal contribution** |
| **DGCNN + Temporal Transformer (15D concat)** | 15D | Transformer | **NEW: critical control — same temporal head, no FiLM/AttnRes** |
| **DGCNN-AFTNet v2 (full)** | 6D point + 9D frame | Transformer + FiLM + AttnRes | **Proposed model** |

The two new DGCNN+Transformer baselines directly address reviewer Xiwx and fJX4's requests. They isolate the contribution of the conditioning mechanisms from the temporal modeling.

### Notes on PST2 / P4Transformer (reviewer fJX4)
- Evaluate feasibility of including these. If not feasible (e.g., incompatible input format for sparse radar), justify exclusion in paper with specific technical reasons.

---

## 5. Ablation Design

### 5A. Component ablation (extends Table 3 from v1)

| Variant | Point FiLM | Frame AttnRes | Temporal Transformer | Pos. Embed | Stride |
|---------|-----------|---------------|---------------------|-----------|--------|
| Full model | yes | yes | yes | yes | s=10 |
| w/o Frame-level AttnRes | yes | **no** | yes | yes | s=10 |
| w/o Point-level FiLM | **no (concat)** | yes | yes | yes | s=10 |
| w/o Both conditioning | **no** | **no** | yes | yes | s=10 |
| w/o Transformer Encoder | yes | yes | **no (mean pool)** | n/a | s=10 |
| w/o Positional Encoding | yes | yes | yes | **no** | s=10 |
| w/o Stride (dense s=1) | yes | yes | yes | yes | **s=1** |

### 5B. Per-feature ablation (NEW — directly requested by reviewers)

| Variant | Point-level aux | Frame-level aux |
|---------|----------------|-----------------|
| Full model | Doppler, SNR, Density | IMU, BLE |
| w/o Doppler | SNR, Density | IMU, BLE |
| w/o SNR | Doppler, Density | IMU, BLE |
| w/o Density | Doppler, SNR | IMU, BLE |
| w/o IMU | Doppler, SNR, Density | BLE |
| w/o BLE | Doppler, SNR, Density | IMU |
| w/o all frame-level aux | Doppler, SNR, Density | — |
| w/o all point-level aux | — | IMU, BLE |
| XYZ only (no aux at all) | — | — |

### 5C. Frame-level conditioning mechanism ablation (NEW — validates AttnRes choice)

| Mechanism | Description |
|-----------|-------------|
| No frame conditioning | Frame-level aux not used |
| FiLM (frame-level) | MLP(IMU,BLE) → gamma, beta → scale+shift frame embedding |
| Simple cross-attention | Single-layer cross-attention, frame embedding attends to single aux vector |
| AttnRes-style (proposed) | Multi-level MLP stack + depth-selective attention |

This table directly shows why AttnRes-style conditioning is better than simpler alternatives for frame-level signals.

---

## 6. Additional Reporting (reviewer requests)

### Per-class F1 scores
- Report macro F1 for all 30 classes, aggregated across all LOSO folds
- Include per-class breakdown in appendix (table or bar chart)

### Aggregate confusion matrix
- Average confusion matrix across all 20 LOSO folds (not just Subject 8)
- Highlight high-risk misclassifications between risk categories

### Per-fold performance
- Report accuracy and F1 for each of the 20 subject folds
- Show distribution via boxplot (as in v1) plus individual fold table in appendix

### Clinically relevant error metrics
- Per-risk-level precision, recall, F1 across all folds
- Specifically report high-risk category (L4) error rates

---

## 7. Paper Writing Changes

### Framing
- **Drop:** "modality-agnostic" as main framing → replace with "multi-granularity auxiliary conditioning"
- **New narrative:** Two-tier conditioning framework: point-level FiLM for per-point sensor attributes + frame-level AttnRes-inspired attention for subject-level wearable/environmental signals
- **Contribution shift:** From "FiLM on radar aux" to "a unified framework that handles auxiliary signals at both point and frame granularities, demonstrated across radar, inertial, and RF modalities"

### Zone discussion
- Remove zone from auxiliary features and all experiments
- Add to limitations: "Our evaluation is conducted in a single simulated room. Future work should investigate learned spatial context representations and cross-environment generalization."

### Limitations section (MUST ADD)
1. Single simulated room, fixed radar position — no cross-room validation
2. Healthy participants, not real patients or caregivers
3. Privacy improvement over RGB is relative, not absolute
4. Clinical deployment requires human-in-the-loop safeguards
5. Demographic and environmental bias not fully characterized

### Impact statement (MUST ADD)
- Positive: privacy-preserving monitoring, infection prevention
- Risks: surveillance concerns, false positive/negative consequences for clinical decisions
- Mitigation: human-in-the-loop design, not fully automated decision-making

### Fix inconsistencies
- Unify k-NN to single value (decide: k=20 given N_max=22)
- Fix "molality" → "modality", "axillary" → "auxiliary"
- Clean up notation (E vs Z for temporal embeddings)
- Clarify preprocessing: how N_max=22 is chosen, padding strategy, train/val split protocol

---

## 8. Implementation Roadmap

### Phase 1: Data pipeline
1. Modify `misc/carelab_extractor.py` to extract IMU (6D) and BLE (3D) signals
2. Align IMU/BLE timestamps with radar frame timestamps
3. Store frame-level signals alongside point-level data in pickle output
4. Update `mmrnet/dataset/mmrnet_data.py` (`MMRActionData`) to load and serve frame-level signals
5. Remove zone ID from extraction and dataset loading
6. Generate new processed data files

### Phase 2: Model architecture
7. Create new model file `dgcnn_aux_fusion_t_v2.py` (or modify existing)
8. Add frame-level AttnRes conditioning module after global max pool
9. Add ablation flags: `use_frame_attnres`, `frame_aux_dim`, individual feature toggles
10. Register new model in `mmrnet/models/__init__.py`
11. Update CLI argument parsing in `mmrnet/cli.py` for new flags

### Phase 3: Baselines
12. Modify baseline models to accept 15D input (broadcast frame-level to points)
13. Create `dgcnn_temporal_concat` baseline (DGCNN + Transformer, no FiLM/AttnRes, 15D concat)
14. Create `dgcnn_temporal_3d` baseline (DGCNN + Transformer, XYZ only)
15. Investigate PST2 / P4Transformer feasibility

### Phase 4: Experiments
16. Run all baselines with LOSO (20 folds each)
17. Run full ablation matrix (component + per-feature + mechanism)
18. Collect per-class F1, aggregate confusion matrices, per-fold tables
19. Statistical significance tests (Wilcoxon signed-rank)

### Phase 5: Paper
20. Rewrite methodology section with two-tier conditioning
21. Add limitations and impact statement
22. Update all tables and figures
23. Fix all typos and notation inconsistencies
24. Proofread

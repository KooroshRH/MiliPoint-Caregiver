# NeurIPS 2026 Revision Plan: DGCNN-MMC-T (Multi-Modal Conditioning + Temporal)

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

### 3.2 Frame-level conditioning: Cross-modal attention (NEW)

After global max-pooling produces frame embeddings `E_t ∈ R^D`, condition them on IMU+BLE via cross-modal attention where the radar representation queries the auxiliary signals.

**Why cross-attention (not FiLM) for frame-level?**
FiLM is correct at the point level because Doppler/SNR/Density are *co-located* with each point — the modulation is local and direct. IMU and BLE are foreign-modality signals that describe the subject's global state. The right operation is content-dependent lookup: "given what the radar captured this frame, which aspects of the subject's motion/position are relevant?" This selective query-based integration is what attention is designed for. FiLM would impose a fixed scale+shift regardless of what the radar captured.

**Why not cross-attention at the point level too?**
Point-level signals (Doppler, SNR, Density) vary per-point and are physically co-located with each radar reflection — there is no selection problem, the relationship is direct and local. Cross-attention at that granularity would be unnecessarily expressive and would lose the physical co-location inductive bias.

**Design (lightweight single-head cross-attention):**

```
# Normalize frame signals (handles scale mismatch: acc ≈ m/s², gyro ≈ rad/s, BLE ≈ dBm)
s_t = LayerNorm(frame_signals_t)              # (B, T, 9)

# Project to small cross-attention space
q_t = W_q(E_t)                               # (B, T, d_ca)  radar queries
k_t = W_k(s_t)                               # (B, T, d_ca)  aux keys
v_t = W_v(s_t)                               # (B, T, d_ca)  aux values

# Single-head scaled dot-product attention (per frame, not across time)
alpha_t = softmax(q_t * k_t / sqrt(d_ca))    # (B, T, d_ca) — element-wise, lightweight
ctx_t   = alpha_t * v_t                      # (B, T, d_ca)

# Project back and residual connection
E'_t = E_t + W_o(ctx_t)                      # (B, T, D)
```

`d_ca = 64` (small projection to keep it lightweight given 9-dim input).

**Ablation flags:** `use_frame_conditioning` (disables cross-attention entirely) and `use_film_modulation` (disables point-level FiLM) are independent — enabling full 2x2 ablation.

**Reviewer defense:** "FiLM is appropriate at the point level because the conditioning signal is co-located and the relationship is direct. Cross-attention is appropriate at the frame level because IMU and BLE are foreign-modality context signals — the content-dependence of attention is necessary when the conditioning signal comes from a different physical sensor."

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
- **New narrative:** Two-tier conditioning framework: point-level FiLM for co-located per-point radar attributes + frame-level cross-modal attention for subject-level wearable/environmental signals (IMU, BLE)
- **Contribution shift:** From "FiLM on radar aux" to "a principled two-level conditioning hierarchy — FiLM where signals are co-located and local, cross-attention where signals are foreign-modality and global"
- **Model name:** DGCNN-MMC-T (Multi-Modal Conditioning + Temporal)
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
7. Create new model file `dgcnn_mmc_t.py` (Multi-Modal Conditioning + Temporal)
8. Add frame-level cross-modal attention module (FrameCrossAttn) after global max pool
9. Add ablation flags: `use_frame_conditioning` (frame-level cross-attn), `use_film_modulation` (point-level FiLM) — independent 2x2 ablation
10. Register as `dgcnn_mmc_t` in `mmrnet/models/__init__.py`
11. Update CLI argument parsing in `mmrnet/cli.py` for new flags
12. Keep `dgcnn_aux_fusion_t` unchanged for backward compatibility with existing checkpoints

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

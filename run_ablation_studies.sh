#!/bin/bash

################################################################################
# Ablation Studies for DGCNNAuxFusionT Model
#
# Configuration:
# - Cross-validation: LOSO (Leave-One-Subject-Out)
# - Subjects: 1, 2, 3, 4, 5
# - Learning rate: 1e-3
# - Sampling rate: 10
# - Stacks: 39
# - Temporal format: enabled
#
# Experiments:
# 1. w/o Temporal Transformer (--model-temporal-layers 0)
# 2. w/o Auxiliary Modulation (--model-no-film-modulation)
# 3. w/o Learnable Pos. Embed (--model-no-temporal-pos-embed)
# 4. w/o Auxiliary Features (--model-aux-dim 0)
################################################################################

# Common parameters for all experiments
SUBJECTS="1,2,3,4,5"
MODEL="dgcnn_aux_fusion_t"
CV="LOSO"
LR="1e-3"
SAMPLING_RATE="10"
STACKS="39"

echo "========================================================================"
echo "DGCNNAuxFusionT Ablation Studies"
echo "========================================================================"
echo "Subjects: ${SUBJECTS}"
echo "Configuration: LOSO CV, LR=${LR}, sampling_rate=${SAMPLING_RATE}, stacks=${STACKS}"
echo ""
echo "Note: Baseline results already available. Running 4 ablation studies."
echo "========================================================================"
echo ""

################################################################################
# Ablation 1: w/o Temporal Transformer
################################################################################

echo "========================================================================"
echo "ABLATION 1: w/o Temporal Transformer"
echo "========================================================================"
echo "Setting: --model-temporal-layers 0"
echo "Effect: Disables transformer encoder, uses simple mean pooling over time"
echo ""

python runner.py \
    --model ${MODEL} \
    --cross-validation ${CV} \
    --subject-id ${SUBJECTS} \
    -lr ${LR} \
    --sampling-rate ${SAMPLING_RATE} \
    --stacks ${STACKS} \
    --use-temporal-format \
    --model-temporal-layers 0

echo ""
echo "Ablation 1 jobs submitted!"
echo ""

################################################################################
# Ablation 2: w/o Auxiliary Modulation (FiLM)
################################################################################

echo "========================================================================"
echo "ABLATION 2: w/o Auxiliary Modulation (FiLM)"
echo "========================================================================"
echo "Setting: --model-no-film-modulation"
echo "Effect: Disables gamma/beta modulation in EdgeConvAuxLayer"
echo ""

python runner.py \
    --model ${MODEL} \
    --cross-validation ${CV} \
    --subject-id ${SUBJECTS} \
    -lr ${LR} \
    --sampling-rate ${SAMPLING_RATE} \
    --stacks ${STACKS} \
    --use-temporal-format \
    --model-no-film-modulation

echo ""
echo "Ablation 2 jobs submitted!"
echo ""

################################################################################
# Ablation 3: w/o Learnable Positional Embeddings
################################################################################

echo "========================================================================"
echo "ABLATION 3: w/o Learnable Positional Embeddings"
echo "========================================================================"
echo "Setting: --model-no-temporal-pos-embed"
echo "Effect: Removes learnable temporal positional embeddings from transformer"
echo ""

python runner.py \
    --model ${MODEL} \
    --cross-validation ${CV} \
    --subject-id ${SUBJECTS} \
    -lr ${LR} \
    --sampling-rate ${SAMPLING_RATE} \
    --stacks ${STACKS} \
    --use-temporal-format \
    --model-no-temporal-pos-embed

echo ""
echo "Ablation 3 jobs submitted!"
echo ""

################################################################################
# Ablation 4: w/o Auxiliary Features
################################################################################

echo "========================================================================"
echo "ABLATION 4: w/o Auxiliary Features"
echo "========================================================================"
echo "Setting: --model-aux-dim 0"
echo "Effect: Uses only XYZ coordinates, ignores auxiliary radar metadata"
echo ""

python runner.py \
    --model ${MODEL} \
    --cross-validation ${CV} \
    --subject-id ${SUBJECTS} \
    -lr ${LR} \
    --sampling-rate ${SAMPLING_RATE} \
    --stacks ${STACKS} \
    --use-temporal-format \
    --model-aux-dim 0

echo ""
echo "Ablation 4 jobs submitted!"
echo ""

################################################################################
# Summary
################################################################################

echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo ""
echo "All ablation study jobs have been submitted!"
echo ""
echo "Jobs submitted:"
echo "  - Ablation 1 (w/o Temporal Transformer): 5 jobs (subjects 1-5)"
echo "  - Ablation 2 (w/o Auxiliary Modulation):  5 jobs (subjects 1-5)"
echo "  - Ablation 3 (w/o Learnable Pos. Embed): 5 jobs (subjects 1-5)"
echo "  - Ablation 4 (w/o Auxiliary Features):    5 jobs (subjects 1-5)"
echo ""
echo "Total: 20 jobs"
echo ""
echo "Checkpoint naming convention:"
echo "  - Ablation 1: *_woTemporal"
echo "  - Ablation 2: *_woFiLM"
echo "  - Ablation 3: *_woPosEmbed"
echo "  - Ablation 4: *_woAux"
echo ""
echo "Use 'squeue -u \$USER' to monitor job status"
echo "========================================================================"

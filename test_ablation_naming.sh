#!/bin/bash

################################################################################
# Test Script: Verify Ablation Study Naming
#
# This script runs the ablation commands in dry-run mode to preview the
# generated experiment names without submitting jobs to SLURM.
################################################################################

echo "========================================================================"
echo "Testing Ablation Study Naming (Dry-Run Mode)"
echo "========================================================================"
echo ""
echo "This will show the generated experiment names for each ablation study"
echo "without actually submitting jobs to SLURM."
echo ""

SUBJECTS="1,2,3,4,5"
MODEL="dgcnn_aux_fusion_t"
CV="LOSO"
LR="1e-3"
SAMPLING_RATE="10"
STACKS="39"

################################################################################
# Test 1: w/o Temporal Transformer
################################################################################

echo "========================================================================"
echo "TEST 1: w/o Temporal Transformer (should have '_woTemporal' postfix)"
echo "========================================================================"

python runner.py \
    --model ${MODEL} \
    --cross-validation ${CV} \
    --subject-id ${SUBJECTS} \
    -lr ${LR} \
    --sampling-rate ${SAMPLING_RATE} \
    --stacks ${STACKS} \
    --use-temporal-format \
    --model-temporal-layers 0 \
    --dry-run 2>&1 | grep -A 3 "Experiment:"

echo ""

################################################################################
# Test 2: w/o Auxiliary Modulation (FiLM)
################################################################################

echo "========================================================================"
echo "TEST 2: w/o Auxiliary Modulation (should have '_woFiLM' postfix)"
echo "========================================================================"

python runner.py \
    --model ${MODEL} \
    --cross-validation ${CV} \
    --subject-id ${SUBJECTS} \
    -lr ${LR} \
    --sampling-rate ${SAMPLING_RATE} \
    --stacks ${STACKS} \
    --use-temporal-format \
    --model-no-film-modulation \
    --dry-run 2>&1 | grep -A 3 "Experiment:"

echo ""

################################################################################
# Test 3: w/o Learnable Positional Embeddings
################################################################################

echo "========================================================================"
echo "TEST 3: w/o Learnable Pos. Embed (should have '_woPosEmbed' postfix)"
echo "========================================================================"

python runner.py \
    --model ${MODEL} \
    --cross-validation ${CV} \
    --subject-id ${SUBJECTS} \
    -lr ${LR} \
    --sampling-rate ${SAMPLING_RATE} \
    --stacks ${STACKS} \
    --use-temporal-format \
    --model-no-temporal-pos-embed \
    --dry-run 2>&1 | grep -A 3 "Experiment:"

echo ""

################################################################################
# Test 4: w/o Auxiliary Features
################################################################################

echo "========================================================================"
echo "TEST 4: w/o Auxiliary Features (should have '_woAux' postfix)"
echo "========================================================================"

python runner.py \
    --model ${MODEL} \
    --cross-validation ${CV} \
    --subject-id ${SUBJECTS} \
    -lr ${LR} \
    --sampling-rate ${SAMPLING_RATE} \
    --stacks ${STACKS} \
    --use-temporal-format \
    --model-aux-dim 0 \
    --dry-run 2>&1 | grep -A 3 "Experiment:"

echo ""

################################################################################
# Test 5: Baseline (no ablation)
################################################################################

echo "========================================================================"
echo "TEST 5: Baseline (should have NO ablation postfix)"
echo "========================================================================"

python runner.py \
    --model ${MODEL} \
    --cross-validation ${CV} \
    --subject-id ${SUBJECTS} \
    -lr ${LR} \
    --sampling-rate ${SAMPLING_RATE} \
    --stacks ${STACKS} \
    --use-temporal-format \
    --dry-run 2>&1 | grep -A 3 "Experiment:"

echo ""

################################################################################
# Test 6: Combined Ablations
################################################################################

echo "========================================================================"
echo "TEST 6: Combined (should have '_woTemporal_woFiLM' postfix)"
echo "========================================================================"

python runner.py \
    --model ${MODEL} \
    --cross-validation ${CV} \
    --subject-id 1 \
    -lr ${LR} \
    --sampling-rate ${SAMPLING_RATE} \
    --stacks ${STACKS} \
    --use-temporal-format \
    --model-temporal-layers 0 \
    --model-no-film-modulation \
    --dry-run 2>&1 | grep -A 3 "Experiment:"

echo ""
echo "========================================================================"
echo "Testing Complete!"
echo "========================================================================"
echo ""
echo "Verify that:"
echo "  1. Test 1 names contain '_woTemporal'"
echo "  2. Test 2 names contain '_woFiLM'"
echo "  3. Test 3 names contain '_woPosEmbed'"
echo "  4. Test 4 names contain '_woAux'"
echo "  5. Test 5 names have NO ablation postfix"
echo "  6. Test 6 names contain '_woTemporal_woFiLM'"
echo ""

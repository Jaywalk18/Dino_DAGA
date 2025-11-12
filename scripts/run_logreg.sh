#!/bin/bash
# Logistic Regression evaluation script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="cifar100"
DATA_PATH="/home/user/zhoutianjian/DataSets/cifar"
BASE_OUTPUT_DIR="outputs/logreg"

# Logistic Regression hyperparameters
BATCH_SIZE=256
INPUT_SIZE=224
NUM_WORKERS=4
SAMPLE_RATIO="1"
MAX_ITER=1000
TOLERANCE=1e-12

# Set dummy values for print_config
EPOCHS="N/A"
LR="N/A"

# Override default GPU if needed - use fewer GPUs for testing
DEFAULT_GPU_IDS="3,4"  # Use 2 GPUs for faster testing

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Logistic Regression Evaluation"

# Run experiments
# Baseline - pretrained weights
# run_experiment "main_logreg.py" "01_baseline" "Baseline (No DAGA)" \
#     --max_iter $MAX_ITER --tolerance $TOLERANCE

# DAGA with fine-tuned weights
PRETRAINED_PATH="/home/user/zhoutianjian/Dino_DAGA/outputs/classification/04_daga_hourglass_layer/cifar100_daga_L1-2-10-11_2025-11-11/best_model.pth" \
run_experiment "main_logreg.py" "03_daga_hourglass" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11 \
    --max_iter $MAX_ITER --tolerance $TOLERANCE

echo -e "\n🎉 Logistic Regression evaluation completed!"


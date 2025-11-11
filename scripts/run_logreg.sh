#!/bin/bash
# Logistic Regression evaluation script following official DINOv3 configuration
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
NUM_WORKERS=8
SAMPLE_RATIO=""  # Empty = use full dataset
MAX_ITER=1000
TOLERANCE=1e-12

# Set dummy values for print_config (LogReg uses different params)
EPOCHS="N/A"
LR="N/A"

# Override default GPU if needed
# DEFAULT_GPU_IDS="1,2,3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Logistic Regression Evaluation"

# Run experiments
# Baseline model - no DAGA
run_experiment "main_logreg.py" "01_baseline" "Baseline (No DAGA)" \
    --max_iter $MAX_ITER --tolerance $TOLERANCE

# DAGA last layer
run_experiment "main_logreg.py" "02_daga_last_layer" "DAGA Single Layer (L11)" \
    --use_daga --daga_layers 11 \
    --max_iter $MAX_ITER --tolerance $TOLERANCE

# DAGA hourglass layers
run_experiment "main_logreg.py" "03_daga_hourglass" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11 \
    --max_iter $MAX_ITER --tolerance $TOLERANCE

echo -e "\n🎉 Logistic Regression evaluation completed!"


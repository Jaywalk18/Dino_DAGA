#!/bin/bash
# Linear probe evaluation script following official DINOv3 configuration
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="cifar100"
DATA_PATH="/home/user/zhoutianjian/DataSets/cifar"
BASE_OUTPUT_DIR="outputs/linear"

# Linear probe hyperparameters
BATCH_SIZE=128
INPUT_SIZE=224
NUM_WORKERS=8
SAMPLE_RATIO=""  # Empty = use full dataset
LINEAR_EPOCHS=10
EPOCH_LENGTH=1250
LEARNING_RATES="1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2"

# Override default GPU if needed
# DEFAULT_GPU_IDS="1,2,3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Linear Probe Evaluation"

# Run experiments
# Baseline model - no DAGA
run_experiment "main_linear.py" "01_baseline" "Baseline (No DAGA)" \
    --linear_epochs $LINEAR_EPOCHS --epoch_length $EPOCH_LENGTH \
    --learning_rates $LEARNING_RATES

# DAGA last layer
run_experiment "main_linear.py" "02_daga_last_layer" "DAGA Single Layer (L11)" \
    --use_daga --daga_layers 11 \
    --linear_epochs $LINEAR_EPOCHS --epoch_length $EPOCH_LENGTH \
    --learning_rates $LEARNING_RATES

# DAGA hourglass layers
run_experiment "main_linear.py" "03_daga_hourglass" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11 \
    --linear_epochs $LINEAR_EPOCHS --epoch_length $EPOCH_LENGTH \
    --learning_rates $LEARNING_RATES

echo -e "\n🎉 Linear probe evaluation completed!"


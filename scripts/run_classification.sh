#!/bin/bash
# Classification training script
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load common configuration
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="cifar100"
DATA_PATH="/home/user/zhoutianjian/DataSets/cifar"
BASE_OUTPUT_DIR="outputs/classification"

# Training hyperparameters (optimized for classification)
EPOCHS=2
BATCH_SIZE=32  # Per-GPU batch size (classification can use larger batches)
INPUT_SIZE=224
LR=1e-3
SAMPLE_RATIO=""  # Empty = use full dataset
LOG_FREQ=10

# Override default GPU if needed
# DEFAULT_GPU_IDS="3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Classification"

# Run experiments
run_experiment "main_classification.py" "01_baseline" "Baseline"

run_experiment "main_classification.py" "02_daga_last_layer" "DAGA Single Layer (L11)" \
    --use_daga --daga_layers 11

echo -e "\n🎉 Classification training completed!"

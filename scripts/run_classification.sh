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
DATASET="imagenet"
DATA_PATH="/home/user/zhoutianjian/DataSets/imagenet"
BASE_OUTPUT_DIR="outputs/classification"

# Training hyperparameters (matching raw_code for optimal performance)
EPOCHS=20
BATCH_SIZE=256
INPUT_SIZE=224
LR=5e-1
NUM_WORKERS=16  # Data loading workers
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



# Alternative: visualize middle layer to see early DAGA effects clearly
# run_experiment "main_classification.py" "03_daga_hourglass_vis_early" "DAGA Four Layers, Vis at L5" \
#     --use_daga --daga_layers 1 2 10 11 --vis_attn_layer 11

run_experiment "main_classification.py" "04_daga_hourglass_layer" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11

echo -e "\n🎉 Classification training completed!"

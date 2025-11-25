#!/bin/bash
# Segmentation training script following official DINOv3 configuration
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load common configuration
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="ade20k"
DATA_PATH="/home/user/zhoutianjian/DataSets/ADE20K_2021_17_01"
BASE_OUTPUT_DIR="outputs/segmentation"

# Training hyperparameters (optimized for segmentation)
EPOCHS=50
BATCH_SIZE=16  # Per-GPU batch size (segmentation needs more memory)
INPUT_SIZE=518
LR=5e-3  # Higher LR for segmentation (official uses 1e-3)
NUM_WORKERS=16  # Moderate workers for segmentation
SAMPLE_RATIO=""  # Use 1% of data for quick test (set to empty for full dataset)
OUT_INDICES="2 5 8 11"  # Multi-layer features for segmentation
LOG_FREQ=5

# Override default GPU if needed
# DEFAULT_GPU_IDS="3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Segmentation"

# Run experiments
run_experiment "main_segmentation.py" "01_baseline" "Baseline (Lightweight Head)"

run_experiment "main_segmentation.py" "02_daga_four_layers" "DAGA Four Layers (L2 L5 L8 L11)" \
    --use_daga --daga_layers 2 5 8 11

echo -e "\n🎉 Segmentation training completed!"

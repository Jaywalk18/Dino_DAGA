#!/bin/bash
# Detection training script following official DINOv3 configuration
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="coco"
DATA_PATH="/home/user/zhoutianjian/DataSets/COCO 2017"
BASE_OUTPUT_DIR="outputs/detection"

# Training hyperparameters
EPOCHS=30
BATCH_SIZE=32
INPUT_SIZE=518
LR=2e-4  
NUM_WORKERS=8  
SAMPLE_RATIO=1
LAYERS_TO_USE="2 5 8 11"
LOG_FREQ=5

# Override default GPU if needed (4 GPUs: 3,4,5,6)
DEFAULT_GPU_IDS="2,3,4,5,6"

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Detection"

# # # Run experiments
# run_experiment "main_detection.py" "01_baseline" "Baseline (Multi-layer Features)"

run_experiment "main_detection.py" "03_daga_detection_four_layers" "DAGA Four Layers (L1 L4 L7 L10)" \
    --use_daga --daga_layers 1 4 7 10

echo -e "\n🎉 Detection training completed!"

#!/bin/bash
# Detection training script following official DINOv3 configuration
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load common configuration
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="coco"
DATA_PATH="/home/user/zhoutianjian/DataSets/COCO 2017"
BASE_OUTPUT_DIR="outputs/detection"

# Training hyperparameters (optimized for detection)
EPOCHS=20
BATCH_SIZE=128  # Per-GPU batch size (detection needs more memory)
INPUT_SIZE=518
LR=3e-2
SAMPLE_RATIO=1  # Use 10% of data for quick test (set to empty for full dataset)
LAYERS_TO_USE="2 5 8 11"  # Multi-layer features like official
LOG_FREQ=1

# Override default GPU if needed
# DEFAULT_GPU_IDS="3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Detection"

# Run experiments
run_experiment "main_detection.py" "01_baseline" "Baseline (Multi-layer Features)"

run_experiment "main_detection.py" "02_daga_hourglass_layer" "DAGA Four Layers (L1 L2 L10 L11)" \
    --use_daga --daga_layers 1 2 10 11

echo -e "\n🎉 Detection training completed!"

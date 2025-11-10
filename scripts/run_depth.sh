#!/bin/bash
# Depth Estimation evaluation script following official DINOv3 configuration
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="nyu_depth_v2"  # or "kitti"
DATA_PATH="/home/user/zhoutianjian/DataSets/NYUDepthV2"
BASE_OUTPUT_DIR="outputs/depth"

# Depth estimation hyperparameters
BATCH_SIZE=16  # Depth estimation needs more memory
INPUT_SIZE=518
NUM_WORKERS=4
SAMPLE_RATIO=""  # Empty = use full dataset

# Depth range configuration
MIN_DEPTH=0.001
MAX_DEPTH=10.0

# Output layers for depth (multi-scale features)
# For ViT-B/L: use layers [4, 11, 17, 23] for 24-layer models
# For ViT-B: use layers [2, 5, 8, 11] for 12-layer models
OUT_INDICES="2 5 8 11"  # For 12-layer models (vitb16, vits16)

# Override default GPU if needed
# DEFAULT_GPU_IDS="1,2,3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Depth Estimation"

# Run experiments
# Baseline model - no DAGA
run_experiment "main_depth.py" "01_baseline" "Baseline (Multi-scale Features)" \
    --min_depth $MIN_DEPTH --max_depth $MAX_DEPTH

# DAGA last layer
run_experiment "main_depth.py" "02_daga_last_layer" "DAGA Single Layer (L11)" \
    --use_daga --daga_layers 11 \
    --min_depth $MIN_DEPTH --max_depth $MAX_DEPTH

# DAGA hourglass layers
run_experiment "main_depth.py" "03_daga_hourglass" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11 \
    --min_depth $MIN_DEPTH --max_depth $MAX_DEPTH

echo -e "\n🎉 Depth estimation evaluation completed!"


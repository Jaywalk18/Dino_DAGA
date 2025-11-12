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
BATCH_SIZE=4  
INPUT_SIZE=518
NUM_WORKERS=8
SAMPLE_RATIO=""  # Empty = use full dataset (or "0.1" for testing with 10%)
EPOCHS=100  
LR=2e-4  

# Visualization and logging
LOG_FREQ=1  
NUM_VIS_SAMPLES=4  # Number of samples to visualize

# Depth range configuration
MIN_DEPTH=0.001
MAX_DEPTH=10.0

# Output layers for depth (multi-scale features)
# For ViT-B/L: use layers [4, 11, 17, 23] for 24-layer models
# For ViT-B: use layers [2, 5, 8, 11] for 12-layer models
OUT_INDICES="2 5 8 11"  

# Override default GPU if needed
# DEFAULT_GPU_IDS="2,3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Depth Estimation"

# Run experiments
# Baseline model - no DAGA
# run_experiment "main_depth.py" "01_baseline" "Baseline (Multi-scale Features)" \
#     --min_depth $MIN_DEPTH --max_depth $MAX_DEPTH

# DAGA on all feature extraction layers
run_experiment "main_depth.py" "02_daga_feature_layers" "DAGA Feature Layers (L2,L5,L8,L11)" \
    --use_daga --daga_layers 2 5 8 11 \
    --min_depth $MIN_DEPTH --max_depth $MAX_DEPTH

echo -e "\n🎉 Depth estimation evaluation completed!"


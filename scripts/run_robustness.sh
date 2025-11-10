#!/bin/bash
# ImageNet-C Robustness evaluation script following official DINOv3 configuration
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="imagenet_c"
# ImageNet-C dataset path (contains 15 corruption types, 5 severity levels each)
DATA_PATH="/home/user/zhoutianjian/DataSets/ImageNet-C"
BASE_OUTPUT_DIR="outputs/robustness"

# Robustness evaluation hyperparameters
BATCH_SIZE=256
INPUT_SIZE=224
NUM_WORKERS=8

# Corruption types to evaluate (all 15 types)
CORRUPTION_TYPES="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"

# Severity levels (1-5, where 5 is most severe)
SEVERITY_LEVELS="1 2 3 4 5"

# Override default GPU if needed
# DEFAULT_GPU_IDS="1,2,3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "ImageNet-C Robustness Evaluation"

echo ""
echo "Corruption Types: ${CORRUPTION_TYPES}"
echo "Severity Levels: ${SEVERITY_LEVELS}"
echo ""

# Run experiments
# Baseline model - no DAGA
run_experiment "main_robustness.py" "01_baseline" "Baseline (No DAGA)" \
    --corruption_types $CORRUPTION_TYPES --severity_levels $SEVERITY_LEVELS

# DAGA last layer
run_experiment "main_robustness.py" "02_daga_last_layer" "DAGA Single Layer (L11)" \
    --use_daga --daga_layers 11 \
    --corruption_types $CORRUPTION_TYPES --severity_levels $SEVERITY_LEVELS

# DAGA hourglass layers
run_experiment "main_robustness.py" "03_daga_hourglass" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11 \
    --corruption_types $CORRUPTION_TYPES --severity_levels $SEVERITY_LEVELS

echo -e "\n🎉 Robustness evaluation completed!"


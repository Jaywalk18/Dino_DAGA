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
DATA_PATH="/home/user/zhoutianjian/DataSets/ImageNet-C/extracted"
BASE_OUTPUT_DIR="outputs/robustness"

# Robustness evaluation hyperparameters
BATCH_SIZE=256
INPUT_SIZE=224
NUM_WORKERS=8

# Full evaluation uses all 15 corruption types:
# gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur
# snow frost fog brightness contrast elastic_transform pixelate jpeg_compression
CORRUPTION_TYPES="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"

# Full evaluation across all severity levels
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

# Trained checkpoint path (IMPORTANT: must use ImageNet-trained model!)
# Update these paths after running classification training
BASELINE_TRAINED_CHECKPOINT="outputs/classification/01_baseline/imagenet_baseline_L_2025-11-21/best_model.pth"
DAGA_TRAINED_CHECKPOINT="outputs/classification/04_daga_hourglass_layer/imagenet_daga_L1-2-10-11_2025-11-21/best_model.pth"

# Run experiments
echo -e "\n📝 Running full robustness evaluation"
echo -e "   - Corruption types: ${CORRUPTION_TYPES}"
echo -e "   - Severity levels: ${SEVERITY_LEVELS}\n"

# Baseline model - no DAGA
# Uncomment after training classification baseline model
run_experiment "main_robustness.py" "01_baseline" "Baseline (No DAGA)" \
    --trained_checkpoint "$BASELINE_TRAINED_CHECKPOINT" \
    --corruption_types $CORRUPTION_TYPES --severity_levels $SEVERITY_LEVELS

# DAGA experiments
# Uncomment after training classification DAGA model
run_experiment "main_robustness.py" "03_daga_hourglass" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11 \
    --trained_checkpoint "$DAGA_TRAINED_CHECKPOINT" \
    --corruption_types $CORRUPTION_TYPES --severity_levels $SEVERITY_LEVELS

echo -e "\n⚠️  Robustness evaluation requires trained classification models!"
echo -e "   Steps:"
echo -e "   1. Train classification models first: bash scripts/run_classification.sh"
echo -e "   2. Update checkpoint paths above"
echo -e "   3. Uncomment the experiment lines you want to run"

echo -e "\n🎉 Robustness evaluation completed!"


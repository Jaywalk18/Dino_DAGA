#!/bin/bash
# Logistic Regression evaluation script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="imagenet"
DATA_PATH="/home/user/zhoutianjian/DataSets/imagenet"
BASE_OUTPUT_DIR="outputs/logreg"

# Logistic Regression hyperparameters
# Note: These parameters are the same for both CIFAR and ImageNet (official DINOv3 config)
BATCH_SIZE=256
INPUT_SIZE=224
NUM_WORKERS=4
SAMPLE_RATIO="1"  # Use full dataset for best results
MAX_ITER=1000  # Max iterations per C value (sklearn lbfgs solver)
TOLERANCE=1e-12  # Convergence tolerance

# Performance notes:
# - Grid searches 45 C values: C = 10^(-6) to 10^5 (hardcoded in dinov3)
# - Trains on CPU using sklearn (no GPU support)
# - Estimated time: 30-60 minutes for CIFAR, 60-120 minutes for ImageNet
# - Uses all CPU cores (n_jobs=-1)
# - Progress bar shows training progress for each C value

# Set dummy values for print_config
EPOCHS="N/A"
LR="N/A"

# Override default GPU if needed - use fewer GPUs for testing
DEFAULT_GPU_IDS="3,4"  # Use 2 GPUs for faster testing

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Logistic Regression Evaluation"

# Run experiments
# Baseline - pretrained weights
run_experiment "main_logreg.py" "01_baseline" "Baseline (No DAGA)" \
    --max_iter $MAX_ITER --tolerance $TOLERANCE

# DAGA with fine-tuned weights
# ⚠️ IMPORTANT: Checkpoint and dataset should match!
# - For CIFAR100: use CIFAR100-trained checkpoint
# - For ImageNet: use ImageNet-trained checkpoint (current setting)
# Current: Using ImageNet-trained DAGA on CIFAR100 (cross-dataset, may not show improvement)
PRETRAINED_PATH="/home/user/zhoutianjian/Dino_DAGA/outputs/classification/04_daga_hourglass_layer/imagenet_daga_L1-2-10-11_2025-11-21/best_model.pth"
run_experiment "main_logreg.py" "03_daga_hourglass" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11 \
    --max_iter $MAX_ITER --tolerance $TOLERANCE

echo -e "\n🎉 Logistic Regression evaluation completed!"


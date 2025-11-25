#!/bin/bash
# Linear probe evaluation script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="imagenet"
DATA_PATH="/home/user/zhoutianjian/DataSets/imagenet"
BASE_OUTPUT_DIR="outputs/linear"

# Linear probe hyperparameters
BATCH_SIZE=128
INPUT_SIZE=224
NUM_WORKERS=8
SAMPLE_RATIO=""  # Empty = use full dataset, or set to "0.1" for 10% subset
LINEAR_EPOCHS=1  # Number of training epochs (1 epoch ≈ 2000 iterations for ImageNet)
EPOCH_LENGTH=500  # Iterations per epoch (was 100, too short!)
LEARNING_RATES="1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1"  # Grid search over 10 LRs

# Set dummy values for print_config
EPOCHS="$LINEAR_EPOCHS"
LR="grid_search"

# Override default GPU if needed
# DEFAULT_GPU_IDS="1,2,3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Linear Probe Evaluation"

# Run experiments
# Baseline - pretrained weights
run_experiment "main_linear.py" "01_baseline" "Baseline (No DAGA)" \
    --linear_epochs $LINEAR_EPOCHS --epoch_length $EPOCH_LENGTH \
    --learning_rates $LEARNING_RATES

# DAGA with fine-tuned weights
# ⚠️ IMPORTANT: Checkpoint and dataset should match for best results!
# - For CIFAR100: use outputs/classification/.../cifar100_daga_L1-2-10-11_xxx/best_model.pth
# - For ImageNet: use outputs/classification/.../imagenet_daga_L1-2-10-11_xxx/best_model.pth
# Current: Using ImageNet-trained DAGA (update path if using CIFAR)
PRETRAINED_PATH="/home/user/zhoutianjian/Dino_DAGA/outputs/classification/04_daga_hourglass_layer/imagenet_daga_L1-2-10-11_2025-11-21/best_model.pth"
run_experiment "main_linear.py" "03_daga_hourglass" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11 \
    --linear_epochs $LINEAR_EPOCHS --epoch_length $EPOCH_LENGTH \
    --learning_rates $LEARNING_RATES

echo -e "\n🎉 Linear probe evaluation completed!"


#!/bin/bash
# KNN evaluation script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
DATASET="cifar100"
DATA_PATH="/home/user/zhoutianjian/DataSets/cifar"
BASE_OUTPUT_DIR="outputs/knn"

# KNN hyperparameters
BATCH_SIZE=256
INPUT_SIZE=224
NUM_WORKERS=8
SAMPLE_RATIO="1"
KNN_K_VALUES="10 20 50 100"
TEMPERATURE=0.07

# Set dummy values for print_config (KNN uses different params)
EPOCHS="N/A"
LR="N/A"

# Override default GPU if needed
# DEFAULT_GPU_IDS="1,2,3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "KNN Evaluation"

# Run experiments
# Baseline - pretrained weights
run_experiment "main_knn.py" "01_baseline" "Baseline (No DAGA)" \
    --knn_k_values $KNN_K_VALUES --temperature $TEMPERATURE

# DAGA with fine-tuned weights
PRETRAINED_PATH="/home/user/zhoutianjian/Dino_DAGA/outputs/classification/04_daga_hourglass_layer/cifar100_daga_L1-2-10-11_2025-11-11/best_model.pth" \
run_experiment "main_knn.py" "03_daga_hourglass" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11 \
    --knn_k_values $KNN_K_VALUES --temperature $TEMPERATURE

echo -e "\n🎉 KNN evaluation completed!"


#!/bin/bash
# DINOtxt (Text-Image Alignment) training script following official DINOv3 configuration
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Task-Specific Configuration
# ============================================================================
# DINOtxt requires image-text paired datasets (e.g., COCO Captions, CC3M, etc.)
DATASET="coco_captions"
DATA_PATH="/home/user/zhoutianjian/DataSets/COCO 2017"
BASE_OUTPUT_DIR="outputs/dinotxt"

# DINOtxt hyperparameters
BATCH_SIZE=128
INPUT_SIZE=224
NUM_WORKERS=8
SAMPLE_RATIO=""  # Empty = use full dataset

# Training configuration
EPOCHS=30
LR=1e-4
WARMUP_EPOCHS=5

# Text encoder configuration
TEXT_EMBED_DIM=512
TEXT_NUM_LAYERS=12
TEXT_NUM_HEADS=8

# Loss weights
CLIP_LOSS_WEIGHT=1.0
GRAM_LOSS_WEIGHT=0.0  # Optional: can enable Gram loss for better alignment

# Override default GPU if needed
# DEFAULT_GPU_IDS="1,2,3,4,5,6"  # Uncomment to override default

# ============================================================================
# Main Execution
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "DINOtxt (Text-Image Alignment)"

echo ""
echo "Text Encoder Config:"
echo "  Embed Dim:  ${TEXT_EMBED_DIM}"
echo "  Layers:     ${TEXT_NUM_LAYERS}"
echo "  Heads:      ${TEXT_NUM_HEADS}"
echo ""

# Run experiments
# Baseline model - no DAGA (frozen vision encoder)
run_experiment "main_dinotxt.py" "01_baseline" "Baseline (Frozen Vision Encoder)" \
    --text_embed_dim $TEXT_EMBED_DIM \
    --text_num_layers $TEXT_NUM_LAYERS \
    --text_num_heads $TEXT_NUM_HEADS \
    --clip_loss_weight $CLIP_LOSS_WEIGHT

# DAGA last layer (frozen but with DAGA)
run_experiment "main_dinotxt.py" "02_daga_last_layer" "DAGA Single Layer (L11)" \
    --use_daga --daga_layers 11 \
    --text_embed_dim $TEXT_EMBED_DIM \
    --text_num_layers $TEXT_NUM_LAYERS \
    --text_num_heads $TEXT_NUM_HEADS \
    --clip_loss_weight $CLIP_LOSS_WEIGHT

# DAGA hourglass layers
run_experiment "main_dinotxt.py" "03_daga_hourglass" "DAGA Four Layers (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11 \
    --text_embed_dim $TEXT_EMBED_DIM \
    --text_num_layers $TEXT_NUM_LAYERS \
    --text_num_heads $TEXT_NUM_HEADS \
    --clip_loss_weight $CLIP_LOSS_WEIGHT

echo -e "\n🎉 DINOtxt training completed!"


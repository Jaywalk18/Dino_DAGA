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
BATCH_SIZE=256
INPUT_SIZE=224
NUM_WORKERS=16
SAMPLE_RATIO=""  # Empty = use full dataset

# Training configuration (following official DINOv3 DINOtxt config)
EPOCHS=100
LR=1e-3     # Reduced from 7e-4 - more stable for contrastive learning
WARMUP_EPOCHS=10  # Increased warmup for better stability
LOG_FREQ=10 

# Text encoder configuration
TEXT_EMBED_DIM=512
TEXT_NUM_LAYERS=12
TEXT_NUM_HEADS=8

# CLIP pretrained text encoder (recommended for faster convergence)
CLIP_PRETRAINED_PATH="checkpoints/clip_vit_b16.pt"

# Loss weights
CLIP_LOSS_WEIGHT=1.0
GRAM_LOSS_WEIGHT=0.0  # Optional: can enable Gram loss for better alignment

# Visualization configuration
NUM_VIS_SAMPLES=3  # Number of samples to visualize (3 images)

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
# ============================================================================
# Experiment 1: Baseline + CLIP (for comparison)
# ============================================================================
# run_experiment "main_dinotxt.py" "01_baseline_clip_pretrained" "Baseline + CLIP Text Encoder" \
#     --text_embed_dim $TEXT_EMBED_DIM \
#     --text_num_layers $TEXT_NUM_LAYERS \
#     --text_num_heads $TEXT_NUM_HEADS \
#     --clip_pretrained_path "$CLIP_PRETRAINED_PATH" \
#     --clip_loss_weight $CLIP_LOSS_WEIGHT \
#     --log_freq $LOG_FREQ \
#     --num_vis_samples $NUM_VIS_SAMPLES

# ============================================================================
# Experiment 2: DAGA + CLIP (current issue: DAGA changes feature distribution,
#               but CLIP text encoder expects original features)
# ============================================================================
# run_experiment "main_dinotxt.py" "02_daga_hourglass_clip_pretrained" "DAGA + CLIP Text Encoder" \
#     --use_daga --daga_layers 1 2 10 11 \
#     --text_embed_dim $TEXT_EMBED_DIM \
#     --text_num_layers $TEXT_NUM_LAYERS \
#     --text_num_heads $TEXT_NUM_HEADS \
#     --clip_pretrained_path "$CLIP_PRETRAINED_PATH" \
#     --clip_loss_weight $CLIP_LOSS_WEIGHT \
#     --log_freq $LOG_FREQ \
#     --num_vis_samples $NUM_VIS_SAMPLES

# ============================================================================
# Experiment 3: Baseline + Scratch Text Encoder (NO CLIP, NO DAGA)
# - Pure baseline for fair comparison
# ============================================================================
run_experiment "main_dinotxt.py" "03_baseline_scratch" "Baseline + Scratch Text Encoder" \
    --text_embed_dim $TEXT_EMBED_DIM \
    --text_num_layers $TEXT_NUM_LAYERS \
    --text_num_heads $TEXT_NUM_HEADS \
    --clip_loss_weight $CLIP_LOSS_WEIGHT \
    --log_freq $LOG_FREQ \
    --num_vis_samples $NUM_VIS_SAMPLES

# ============================================================================
# Experiment 4: DAGA + Scratch Text Encoder (NO CLIP, WITH DAGA)
# - Text encoder trains from scratch, can adapt to DAGA-modified features
# - This is the fair comparison to test if DAGA helps
# ============================================================================
run_experiment "main_dinotxt.py" "04_daga_scratch" "DAGA + Scratch Text Encoder" \
    --use_daga --daga_layers 1 2 10 11 \
    --text_embed_dim $TEXT_EMBED_DIM \
    --text_num_layers $TEXT_NUM_LAYERS \
    --text_num_heads $TEXT_NUM_HEADS \
    --clip_loss_weight $CLIP_LOSS_WEIGHT \
    --log_freq $LOG_FREQ \
    --num_vis_samples $NUM_VIS_SAMPLES

echo -e "\n🎉 DINOtxt training completed!"


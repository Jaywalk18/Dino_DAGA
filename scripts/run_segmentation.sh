#!/bin/bash
# Segmentation training script following official DINOv3 configuration
set -e

# Activate environment
source activate dinov3_env

# Configuration
DATASET="ade20k"
MODEL_NAME="dinov3_vits16"
PRETRAINED_PATH="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DATA_PATH="/home/user/zhoutianjian/DataSets/ADE20K_2021_17_01"
BASE_OUTPUT_DIR="outputs/segmentation"
DEFAULT_GPU_IDS="3,4,5,6"
GPU_IDS="${GPU_IDS:-$DEFAULT_GPU_IDS}"
SEED=42

# Training hyperparameters (quick test with subset of data)
EPOCHS=1  # Quick test with 1 epoch
BATCH_SIZE=16  # Smaller batch to fit multi-GPU memory (≈4 per GPU)
INPUT_SIZE=518
LR=1e-3  # Higher LR for segmentation (official uses 1e-3)
SAMPLE_RATIO="${SAMPLE_RATIO:-0.02}"  # Use 2% of data for quick test
OUT_INDICES="2 5 8 11"  # Multi-layer features

# Setup
PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
CHECKPOINT_DIR="/home/user/zhoutianjian/DAGA/checkpoints"
cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p "$BASE_OUTPUT_DIR"
echo "🚀 DINOv3 Segmentation Training"
echo "  Sample %:   ${SAMPLE_RATIO}"
echo "==================================================================="
echo "  Model:      ${MODEL_NAME}"
echo "  GPU IDs:    ${GPU_IDS}"
echo "  Epochs:     ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  LR:         ${LR}"
echo "  Out Layers: ${OUT_INDICES}"
echo "==================================================================="

# Helper function
run_experiment() {
    local exp_name=$1
    local description=$2
    shift 2

    local output_subdir="${BASE_OUTPUT_DIR}/${exp_name}"
    mkdir -p "$output_subdir"

    echo -e "\n▶️  ${description}"

    local sample_args=()
    if [[ -n "$SAMPLE_RATIO" ]]; then
        sample_args+=(--sample_ratio "$SAMPLE_RATIO")
    fi

    CUDA_VISIBLE_DEVICES=$GPU_IDS python main_segmentation.py \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --model_name "$MODEL_NAME" \
        --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --input_size "$INPUT_SIZE" \
        --lr "$LR" \
        --output_dir "$output_subdir" \
        --enable_visualization \
        --num_vis_samples 4 \
        --log_freq 1 \
        --out_indices $OUT_INDICES \
        "${sample_args[@]}" \
        "$@"

    [ $? -eq 0 ] && echo "✅  SUCCESS" || (echo "❌  FAILED" && exit 1)
}

# Experiments
# run_experiment "01_baseline" "Baseline (Lightweight Head)"

run_experiment "02_daga_last_layer" "DAGA Single Layer (L11)" \
    --use_daga --daga_layers 11

echo -e "\n🎉 Segmentation training completed!"

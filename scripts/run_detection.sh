#!/bin/bash
# Detection training script following official DINOv3 configuration
set -e

# Activate environment
source activate dinov3_env

# Configuration
DATASET="coco"
MODEL_NAME="dinov3_vits16"
PRETRAINED_PATH="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DATA_PATH="/home/user/zhoutianjian/DataSets/COCO 2017"
BASE_OUTPUT_DIR="outputs/detection"
DEFAULT_GPU_IDS="3,4,5,6"
GPU_IDS="${GPU_IDS:-$DEFAULT_GPU_IDS}"
SEED=42

# Training hyperparameters (quick test with subset of data)
EPOCHS=40  # Quick test with 1 epoch
BATCH_SIZE=256  # 4 per GPU (6 GPUs) - increased for faster training
INPUT_SIZE=518
LR=1e-4
SAMPLE_RATIO=0.5  # Use 2% of data for quick test
LAYERS_TO_USE="2 5 8 11"  # Multi-layer features like official

# Setup
PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
CHECKPOINT_DIR="/home/user/zhoutianjian/DAGA/checkpoints"
cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p "$BASE_OUTPUT_DIR"
echo "🚀 DINOv3 Detection Training"
echo "  Sample %:   ${SAMPLE_RATIO}"
echo "==================================================================="
echo "  Model:      ${MODEL_NAME}"
echo "  GPU IDs:    ${GPU_IDS}"
echo "  Epochs:     ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  LR:         ${LR}"
echo "  Layers:     ${LAYERS_TO_USE}"
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

    CUDA_VISIBLE_DEVICES=$GPU_IDS python main_detection.py \
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
        --layers_to_use $LAYERS_TO_USE \
        "${sample_args[@]}" \
        "$@"

    [ $? -eq 0 ] && echo "✅  SUCCESS" || (echo "❌  FAILED" && exit 1)
}

# Experiments
run_experiment "01_baseline" "Baseline (Multi-layer Features)"

run_experiment "02_daga_last_layer" "DAGA Single Layer (L11)" \
    --use_daga --daga_layers 11

echo -e "\n🎉 Detection training completed!"

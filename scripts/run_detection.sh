#!/bin/bash
# run_detection.sh
set -e

# --- Configuration ---
DATASET="coco"
MODEL_NAME="dinov3_vits16"
PRETRAINED_PATH="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DATA_PATH="/home/user/zhoutianjian/DataSets/COCO 2017"
BASE_OUTPUT_DIR="outputs/detection"
GPU_IDS=${1:-"1,2,3,4,5,6"}
SEED=42

# --- Hyperparameters ---
EPOCHS=1
BATCH_SIZE=16
INPUT_SIZE=518
LR=1e-4
MAX_SAMPLES=12000  # ~10% of COCO train set (118K images)

# --- Setup ---
PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
CHECKPOINT_DIR="/home/user/zhoutianjian/DAGA/checkpoints"
cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p "$BASE_OUTPUT_DIR"
echo "🚀 Starting DINOv3 Detection on ${DATASET}..."
echo "==================================================================="
echo "  DINOv3 Model:      ${MODEL_NAME}"
echo "  GPU IDs:           ${GPU_IDS}"
echo "  Training Samples:  ${MAX_SAMPLES} (~10% of dataset)"
echo "  Hyperparameters:"
echo "    - Epochs:        ${EPOCHS}"
echo "    - Batch Size:    ${BATCH_SIZE}"
echo "    - Learning Rate: ${LR}"
echo "==================================================================="

# --- Helper function for running experiments ---
run_experiment() {
    local exp_name=$1
    local description=$2
    shift 2

    local output_subdir="${BASE_OUTPUT_DIR}/${exp_name}"
    mkdir -p "$output_subdir"

    echo -e "\n▶️  Running Experiment: ${description}"

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
        --max_samples "$MAX_SAMPLES" \
        --output_dir "$output_subdir" \
        --enable_visualization \
        --num_vis_samples 4 \
        --log_freq 20 \
        "$@"

    if [ $? -eq 0 ]; then
        echo "✅  SUCCESS: ${description}"
    else
        echo "❌  FAILED: ${description}"
        exit 1
    fi
}

# --- Experiment Runs ---

run_experiment \
    "01_baseline" \
    "Baseline (Linear Detection Head on DINOv3)"

run_experiment \
    "02_daga_last_layer" \
    "DAGA Single Layer (L11)" \
    --use_daga \
    --daga_layers 11

echo -e "\n🎉 All detection experiments completed successfully!"

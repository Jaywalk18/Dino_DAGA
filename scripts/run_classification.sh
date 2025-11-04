#!/bin/bash
# run_classification.sh
set -e

# --- Configuration ---
DATASET="cifar100"
MODEL_NAME="dinov3_vits16"
PRETRAINED_PATH="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DATA_PATH="/home/user/zhoutianjian/DataSets/cifar"
BASE_OUTPUT_DIR="outputs/classification"
GPU_IDS=${1:-"1,2,3,4,5,6"}
SEED=42

# --- Hyperparameters ---
EPOCHS=1
BATCH_SIZE=256
INPUT_SIZE=224
LR=4e-3
SUBSET_RATIO=0.15  # Use 15% of dataset for faster testing (was 0.01 = 1%)

# --- Setup ---
PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
CHECKPOINT_DIR="/home/user/zhoutianjian/DAGA/checkpoints"
cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p "$BASE_OUTPUT_DIR"
echo "🚀 Starting DINOv3 Classification on ${DATASET}..."
echo "==================================================================="
echo "  DINOv3 Model:      ${MODEL_NAME}"
echo "  GPU IDs:           ${GPU_IDS}"
echo "  Training Subset:   ${SUBSET_RATIO} (15% of dataset)"
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

    CUDA_VISIBLE_DEVICES=$GPU_IDS python main_classification.py \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --subset_ratio "$SUBSET_RATIO" \
        --model_name "$MODEL_NAME" \
        --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --input_size "$INPUT_SIZE" \
        --output_dir "$output_subdir" \
        --lr "$LR" \
        --enable_visualization \
        --vis_indices 0 1 2 3 \
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
    "Baseline (Linear Probe on DINOv3)"

run_experiment \
    "02_daga_last_layer" \
    "DAGA Single Layer (L11)" \
    --use_daga \
    --daga_layers 11

echo -e "\n🎉 All classification experiments completed successfully!"

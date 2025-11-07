#!/bin/bash
# run_dinov3_experiments_pth.sh
set -e

# --- Configuration ---
DATASET="cifar100"
MODEL_NAME="dinov3_vitb16"
PRETRAINED_PATH="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DATA_PATH="../DataSets/cifar/"
BASE_OUTPUT_DIR="experiments_dinov3_ablation"
GPU_IDS=${1:-"1,2,3,4,5,6"}
SEED=42

# --- Hyperparameters (aligned with successful baseline) ---
EPOCHS=20
BATCH_SIZE=256
INPUT_SIZE=224
LR=4e-3
SUBSET_RATIO=1

# --- Setup ---
mkdir -p "$BASE_OUTPUT_DIR"
echo "🚀 Starting DINOv3 Ablation Study on ${DATASET}..."
echo "==================================================================="
echo "  DINOv3 Model:      ${MODEL_NAME}"
echo "  Pretrained Path:   ${PRETRAINED_PATH}"
echo "  GPU IDs:           ${GPU_IDS}"
echo "  Seed for training: ${SEED}"
echo "  Training Subset:   ${SUBSET_RATIO} (1.0 = full dataset)"
echo "  Output Directory:  ${BASE_OUTPUT_DIR}"
echo "  Hyperparameters:"
echo "    - Epochs:        ${EPOCHS}"
echo "    - Batch Size:    ${BATCH_SIZE}"
echo "    - Input Size:    ${INPUT_SIZE}"
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

    CUDA_VISIBLE_DEVICES=$GPU_IDS python dinov3_finetune_daga_pth.py \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --subset_ratio "$SUBSET_RATIO" \
        --model_name "$MODEL_NAME" \
        --pretrained_path "$PRETRAINED_PATH" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --input_size "$INPUT_SIZE" \
        --output_dir "$output_subdir" \
        --swanlab_name "${DATASET}_${exp_name}_subset_${SUBSET_RATIO}" \
        --lr "$LR" \
        --enable_visualization \
        --vis_indices 1000 2000 3000 4000 \
        --log_freq 2 \
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
    "01_dinov3_baseline" \
    "Baseline (Linear Probe on DINOv3)"

# run_experiment \
#     "02_dinov3_daga_four_layers" \
#     "DAGA Lightweight" \
#     --use_daga \
#     --daga_layers 2 5 8 11

# run_experiment \
#     "03_dinov3_daga_three_layers" \
#     "DAGA Lightweight" \
#     --use_daga \
#     --daga_layers 3 7 11


# run_experiment \
#     "08_daga_mid_focus" \
#     "DAGA with focus on mid-level layers" \
#     --use_daga \
#     --daga_layers 5 8 11


run_experiment \
    "09_daga_hourglass" \
    "DAGA with hourglass (ends-heavy) distribution" \
    --use_daga \
    --daga_layers 1 2 10 11


# run_experiment \
#     "05_dinov3_daga_balanced" \
#     "DAGA Balanced (last two layers)" \
#     --use_daga \
#     --daga_layers 9 11

echo -e "\n🎉 All DINOv3 experiments completed successfully!"

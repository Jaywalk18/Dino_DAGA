#!/bin/bash
# run_detection.sh
set -e

# =============================================================================
# Section 1: Environment Activation & PYTHONPATH Setup
# =============================================================================
source activate dinov3_env
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$(pwd)

# =============================================================================
# Section 2: Dataset Path Validation
# =============================================================================
DATA_ROOT="/home/user/zhoutianjian/DataSets"
DATASET="coco"
DATA_PATH="${DATA_ROOT}/COCO 2017"

if [ ! -d "$DATA_PATH" ]; then
    echo "❌ ERROR: COCO dataset not found at: $DATA_PATH"
    echo "Please check the data path and try again."
    exit 1
fi

# =============================================================================
# Section 3: Test/Full Mode Switch
# =============================================================================
MODE=${1:-"test"}

if [ "$MODE" = "test" ]; then
    echo "🧪 QUICK TEST MODE"
    MAX_SAMPLES=1200  # ~1% of COCO training data
else
    echo "🚀 FULL TRAINING MODE"
    MAX_SAMPLES=0  # Use all data
fi

# =============================================================================
# Section 4: Hyperparameter Configuration
# =============================================================================

# Model configuration
CHECKPOINT_DIR="/home/user/zhoutianjian/DAGA/checkpoints"
DINOV3_MODEL="dinov3_vits16"
PRETRAINED_PATH="${CHECKPOINT_DIR}/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Training hyperparameters (aligned with COCO detection standards)
if [ "$MODE" = "test" ]; then
    EPOCHS=1
    BATCH_SIZE=16
    LR=1e-4
else
    EPOCHS=12  # Aligned with typical detection training
    BATCH_SIZE=16
    LR=1e-4
fi

SEED=42
INPUT_SIZE=518  # Detection typically uses larger input size
BASE_OUTPUT_DIR="outputs/detection"

# =============================================================================
# Section 5: Output Informative Prompts
# =============================================================================
echo "========================================================================"
echo "🚀 DINOv3 Object Detection - COCO Dataset"
echo "========================================================================"
echo "  Mode:          ${MODE}"
echo "  Model:         ${DINOV3_MODEL}"
echo "  Pretrained:    ${PRETRAINED_PATH}"
echo "  Dataset:       COCO 2017"
echo "  Data Path:     ${DATA_PATH}"
echo "  GPU IDs:       ${CUDA_VISIBLE_DEVICES}"
echo "  Seed:          ${SEED}"
if [ "$MODE" = "test" ]; then
    echo "  Max Samples:   ${MAX_SAMPLES} (~1% data for quick test)"
fi
echo ""
echo "  Hyperparameters:"
echo "    - Epochs:      ${EPOCHS}"
echo "    - Batch Size:  ${BATCH_SIZE}"
echo "    - Input Size:  ${INPUT_SIZE}"
echo "    - Learn Rate:  ${LR}"
echo "  Output Dir:    ${BASE_OUTPUT_DIR}"
echo "========================================================================"
echo ""

# =============================================================================
# Helper function for running experiments
# =============================================================================

run_experiment() {
    local exp_name=$1
    local description=$2
    shift 2
    
    local output_subdir="${BASE_OUTPUT_DIR}/${exp_name}"
    mkdir -p "$output_subdir"
    
    echo -e "\n▶️  Running Experiment: ${description}"
    echo "─────────────────────────────────────────────────────────────────────"
    
    local max_samples_args=""
    if [ "$MAX_SAMPLES" != "0" ]; then
        max_samples_args="--max_samples $MAX_SAMPLES"
    fi
    
    python main_detection.py \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --model_name "$DINOV3_MODEL" \
        --pretrained_path "$PRETRAINED_PATH" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --input_size "$INPUT_SIZE" \
        --lr "$LR" \
        --output_dir "$output_subdir" \
        --swanlab_name "${DATASET}_${exp_name}" \
        --enable_visualization \
        --num_vis_samples 4 \
        --log_freq 2 \
        $max_samples_args \
        "$@"
    
    if [ $? -eq 0 ]; then
        echo "✅  SUCCESS: ${description}"
    else
        echo "❌  FAILED: ${description}"
        exit 1
    fi
}

# =============================================================================
# Experiment Runs
# =============================================================================

run_experiment \
    "01_baseline" \
    "Baseline (Linear Detection Head on DINOv3)"

run_experiment \
    "02_daga_last_layer" \
    "DAGA Single Layer (L11)" \
    --use_daga \
    --daga_layers 11

# Uncomment for multi-layer DAGA experiments
# run_experiment \
#     "03_daga_hourglass" \
#     "DAGA Hourglass Distribution (L1,2,10,11)" \
#     --use_daga \
#     --daga_layers 1 2 10 11

echo ""
echo "========================================================================"
echo "🎉 All COCO Detection Experiments Completed Successfully!"
echo "========================================================================"
echo "Results saved in: ${BASE_OUTPUT_DIR}/"
echo "Check visualizations in: ${BASE_OUTPUT_DIR}/*/visualizations/"
echo "========================================================================"

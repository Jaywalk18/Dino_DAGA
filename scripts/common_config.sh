#!/bin/bash
# Common configuration for all training scripts
# This file should be sourced by run_classification.sh, run_detection.sh, run_segmentation.sh

# ============================================================================
# Signal Handling - Kill all child processes on exit
# ============================================================================
cleanup() {
    echo ""
    echo "🛑 Caught signal, terminating all processes..."
    # Kill all processes in the current process group
    kill -- -$$ 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM SIGHUP

# ============================================================================
# Environment Setup
# ============================================================================
setup_environment() {
    # Activate conda environment
    source activate dinov3_env
    export SWANLAB_MODE=cloud
    
    # DDP Environment Variables for better multi-GPU performance
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=0  # Enable P2P for better performance
    export OMP_NUM_THREADS=1
    export CUDA_LAUNCH_BLOCKING=0
}

# ============================================================================
# Common Configuration
# ============================================================================
# Model settings
MODEL_NAME="${MODEL_NAME:-dinov3_vitb16}"
PRETRAINED_PATH="${PRETRAINED_PATH:-dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"

# Path settings
PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
CHECKPOINT_DIR="/home/user/zhoutianjian/Dino_DAGA/checkpoints"

# GPU Configuration
DEFAULT_GPU_IDS="${DEFAULT_GPU_IDS:-1,2,3,4,5,6}"  
GPU_IDS="${GPU_IDS:-$DEFAULT_GPU_IDS}"

# Training settings
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"  # Number of data loading workers (default: 8)

# ============================================================================
# Helper Functions
# ============================================================================

# Calculate number of GPUs from GPU_IDS
get_num_gpus() {
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    echo ${#GPU_ARRAY[@]}
}

# Setup project paths
setup_paths() {
    cd "$PROJECT_ROOT"
    export PYTHONPATH=$PYTHONPATH:$(pwd)
}

# Print training configuration header
print_config() {
    local task_name=$1
    local num_gpus=$(get_num_gpus)
    
    echo "🚀 DINOv3 ${task_name} Training"
    [[ -n "$SAMPLE_RATIO" ]] && echo "  Sample %:   ${SAMPLE_RATIO}"
    echo "==================================================================="
    echo "  Model:      ${MODEL_NAME}"
    echo "  GPU IDs:    ${GPU_IDS}"
    echo "  Num GPUs:   ${num_gpus}"
    echo "  Epochs:     ${EPOCHS}"
    echo "  Batch Size: ${BATCH_SIZE} per GPU (Total: $((BATCH_SIZE * num_gpus)))"
    echo "  LR:         ${LR}"
    [[ -n "$LAYERS_TO_USE" ]] && echo "  Layers:     ${LAYERS_TO_USE}"
    [[ -n "$OUT_INDICES" ]] && echo "  Out Layers: ${OUT_INDICES}"
    echo "==================================================================="
}

# Generic experiment runner
run_experiment() {
    local main_script=$1
    local exp_name=$2
    local description=$3
    shift 3
    
    local num_gpus=$(get_num_gpus)
    local output_subdir="${BASE_OUTPUT_DIR}/${exp_name}"
    mkdir -p "$output_subdir"
    
    echo -e "\n▶️  ${description}"
    
    # Build sample_args based on script type
    local sample_args=()
    if [[ -n "$SAMPLE_RATIO" ]]; then
        # main_classification.py, main_knn.py, main_linear.py, main_logreg.py use --subset_ratio, others use --sample_ratio
        if [[ "$main_script" == *"classification"* ]] || [[ "$main_script" == *"knn"* ]] || [[ "$main_script" == *"linear"* ]] || [[ "$main_script" == *"logreg"* ]]; then
            sample_args+=(--subset_ratio "$SAMPLE_RATIO")
        else
            sample_args+=(--sample_ratio "$SAMPLE_RATIO")
        fi
    fi
    
    # Build visualization args based on task type
    local vis_args=()
    if [[ "$main_script" == *"classification"* ]]; then
        vis_args+=(--vis_indices 1000 2000 3000 4000)
    elif [[ "$main_script" == *"detection"* ]] || [[ "$main_script" == *"segmentation"* ]] || [[ "$main_script" == *"depth"* ]]; then
        # Use NUM_VIS_SAMPLES variable if set, otherwise default to 4
        vis_args+=(--num_vis_samples "${NUM_VIS_SAMPLES:-4}")
    fi
    
    # Build task-specific args
    local task_args=()
    if [[ -n "$LAYERS_TO_USE" ]]; then
        task_args+=(--layers_to_use $LAYERS_TO_USE)
    fi
    if [[ -n "$OUT_INDICES" ]]; then
        task_args+=(--out_indices $OUT_INDICES)
    fi
    
    # Build training-specific args (not needed for linear/knn/logreg/robustness)
    local training_args=()
    if [[ "$main_script" != *"linear"* ]] && [[ "$main_script" != *"knn"* ]] && [[ "$main_script" != *"logreg"* ]] && [[ "$main_script" != *"robustness"* ]]; then
        training_args+=(--epochs "$EPOCHS")
        training_args+=(--lr "$LR")
        training_args+=(--enable_visualization)
        training_args+=(--log_freq "${LOG_FREQ:-5}")
    fi
    
    # Handle absolute vs relative pretrained paths
    local pretrained_arg
    if [[ "$PRETRAINED_PATH" == /* ]]; then
        # Absolute path - use as is
        pretrained_arg="$PRETRAINED_PATH"
    else
        # Relative path - prepend CHECKPOINT_DIR
        pretrained_arg="${CHECKPOINT_DIR}/${PRETRAINED_PATH}"
    fi
    
    # Use torchrun for DDP training
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$num_gpus \
        "$main_script" \
        --seed "$SEED" \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --model_name "$MODEL_NAME" \
        --pretrained_path "$pretrained_arg" \
        --batch_size "$BATCH_SIZE" \
        --input_size "$INPUT_SIZE" \
        --output_dir "$output_subdir" \
        --num_workers "$NUM_WORKERS" \
        "${training_args[@]}" \
        "${sample_args[@]}" \
        "${vis_args[@]}" \
        "${task_args[@]}" \
        "$@"
    
    [ $? -eq 0 ] && echo "✅  SUCCESS" || (echo "❌  FAILED" && exit 1)
}


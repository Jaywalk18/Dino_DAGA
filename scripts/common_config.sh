#!/bin/bash
# Common configuration for all training scripts
# This file should be sourced by run_classification.sh, run_detection.sh, run_segmentation.sh

# ============================================================================
# Environment Setup
# ============================================================================
setup_environment() {
    # Activate conda environment
    source activate dinov3_env
    # export SWANLAB_MODE=disabled
    
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
PRETRAINED_PATH="${PRETRAINED_PATH:-dinov3_vitb16_pretrain_lvd1689m.pth}"

# Path settings
PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
CHECKPOINT_DIR="/home/user/zhoutianjian/DAGA/checkpoints"

# GPU Configuration
DEFAULT_GPU_IDS="${DEFAULT_GPU_IDS:-3,4,5,6}"  
GPU_IDS="${GPU_IDS:-$DEFAULT_GPU_IDS}"

# Training settings
SEED="${SEED:-42}"

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
        # main_classification.py uses --subset_ratio, others use --sample_ratio
        if [[ "$main_script" == *"classification"* ]]; then
            sample_args+=(--subset_ratio "$SAMPLE_RATIO")
        else
            sample_args+=(--sample_ratio "$SAMPLE_RATIO")
        fi
    fi
    
    # Build visualization args based on task type
    local vis_args=()
    if [[ "$main_script" == *"classification"* ]]; then
        vis_args+=(--vis_indices 1000 2000 3000 4000)
    elif [[ "$main_script" == *"detection"* ]] || [[ "$main_script" == *"segmentation"* ]]; then
        vis_args+=(--num_vis_samples 4)
    fi
    
    # Build task-specific args
    local task_args=()
    if [[ -n "$LAYERS_TO_USE" ]]; then
        task_args+=(--layers_to_use $LAYERS_TO_USE)
    fi
    if [[ -n "$OUT_INDICES" ]]; then
        task_args+=(--out_indices $OUT_INDICES)
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
        --pretrained_path "${CHECKPOINT_DIR}/${PRETRAINED_PATH}" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --input_size "$INPUT_SIZE" \
        --lr "$LR" \
        --output_dir "$output_subdir" \
        --enable_visualization \
        --log_freq "${LOG_FREQ:-5}" \
        "${sample_args[@]}" \
        "${vis_args[@]}" \
        "${task_args[@]}" \
        "$@"
    
    [ $? -eq 0 ] && echo "✅  SUCCESS" || (echo "❌  FAILED" && exit 1)
}


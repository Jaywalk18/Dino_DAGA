#!/bin/bash

# Segmentation Task Script - ADE20K Semantic Segmentation
# Supports both quick test (1 epoch) and full training
# Usage: 
#   Quick test:  bash scripts/run_segmentation.sh test
#   Full train:  bash scripts/run_segmentation.sh

set -e

# =============================================================================
# Configuration
# =============================================================================

# Mode: test or full
MODE=${1:-"full"}

# Environment
source activate dinov3_env
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

# Paths
PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
DATA_ROOT="/home/user/zhoutianjian/DataSets"
CHECKPOINT_DIR="/home/user/zhoutianjian/DAGA/checkpoints"

# Model configuration
DINOV3_MODEL="dinov3_vits16"
PRETRAINED_PATH="${CHECKPOINT_DIR}/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Common parameters
SEED=42
INPUT_SIZE=518
DATASET="ade20k"
DATA_PATH="${DATA_ROOT}/ADE20K_2021_17_01"

cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "========================================================================"
echo "Segmentation Task - DINOv3 Multi-Task Framework"
echo "========================================================================"
echo "Mode: $MODE"
echo "Model: $DINOV3_MODEL"
echo "Dataset: ADE20K"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"
echo ""

# =============================================================================
# Helper function
# =============================================================================

run_experiment() {
    local exp_name=$1
    local epochs=$2
    local batch_size=$3
    local lr=$4
    local use_daga=$5
    local daga_layers=$6
    local swanlab_name=$7
    
    echo "▶️  Running: $exp_name"
    echo "─────────────────────────────────────────────────────────────────────"
    
    local daga_args=""
    if [ "$use_daga" = "true" ]; then
        daga_args="--use_daga --daga_layers $daga_layers"
    fi
    
    python main_segmentation.py \
        --model_name $DINOV3_MODEL \
        --pretrained_path $PRETRAINED_PATH \
        --dataset $DATASET \
        --data_path "$DATA_PATH" \
        --input_size $INPUT_SIZE \
        --epochs $epochs \
        --batch_size $batch_size \
        --lr $lr \
        --seed $SEED \
        $daga_args \
        --out_indices 2 5 8 11 \
        --output_dir ./outputs/segmentation \
        --swanlab_name "$swanlab_name" \
        --enable_visualization \
        --num_vis_samples 4 \
        --log_freq 2
    
    if [ $? -eq 0 ]; then
        echo "✅  SUCCESS: $exp_name"
    else
        echo "❌  FAILED: $exp_name"
        exit 1
    fi
    echo ""
}

# =============================================================================
# Test Mode (1 epoch)
# =============================================================================

if [ "$MODE" = "test" ]; then
    echo "🧪 QUICK TEST MODE (1 epoch)"
    echo ""
    
    # Check if ADE20K dataset exists
    if [ ! -d "$DATA_PATH" ]; then
        echo "❌ ERROR: ADE20K dataset not found at: $DATA_PATH"
        echo "Please check the data path and try again."
        exit 1
    fi
    
    # ADE20K Segmentation Baseline
    run_experiment \
        "ADE20K Segmentation Baseline (Test)" \
        1 \
        16 \
        1e-4 \
        "false" \
        "" \
        "test_ade20k_segmentation_baseline"
    
    # ADE20K Segmentation with DAGA
    run_experiment \
        "ADE20K Segmentation with DAGA (Test)" \
        1 \
        16 \
        1e-4 \
        "true" \
        "11" \
        "test_ade20k_segmentation_daga_L11"
    
    echo "========================================================================"
    echo "✅ QUICK TEST COMPLETED!"
    echo "========================================================================"
    echo "Check outputs in: ./outputs/segmentation/"
    echo "Visualization results saved in: ./outputs/segmentation/*/visualizations/"
    echo "========================================================================"

# =============================================================================
# Full Training Mode
# =============================================================================

else
    echo "🚀 FULL TRAINING MODE"
    echo ""
    
    # Check if ADE20K dataset exists
    if [ ! -d "$DATA_PATH" ]; then
        echo "❌ ERROR: ADE20K dataset not found at: $DATA_PATH"
        echo "Please check the data path and try again."
        exit 1
    fi
    
    # ------------------------------------------------------------------------
    # ADE20K Segmentation Experiments
    # ------------------------------------------------------------------------
    
    echo "========================================================================"
    echo "ADE20K Semantic Segmentation Experiments"
    echo "========================================================================"
    echo ""
    
    # ADE20K Segmentation Baseline
    run_experiment \
        "ADE20K Segmentation Baseline" \
        20 \
        16 \
        1e-4 \
        "false" \
        "" \
        "ade20k_segmentation_baseline"
    
    # ADE20K Segmentation with DAGA (single layer)
    run_experiment \
        "ADE20K Segmentation with DAGA (L11)" \
        20 \
        16 \
        1e-4 \
        "true" \
        "11" \
        "ade20k_segmentation_daga_L11"
    
    # ADE20K Segmentation with DAGA (hourglass) - optional
    # Uncomment if you want to test multi-layer DAGA
    # run_experiment \
    #     "ADE20K Segmentation with DAGA (Hourglass)" \
    #     20 \
    #     16 \
    #     1e-4 \
    #     "true" \
    #     "1 2 10 11" \
    #     "ade20k_segmentation_daga_hourglass"
    
    echo ""
    echo "========================================================================"
    echo "✅ ALL SEGMENTATION EXPERIMENTS COMPLETED!"
    echo "========================================================================"
    echo "Results saved in: ./outputs/segmentation/"
    echo "Visualizations include:"
    echo "  - Attention maps (Frozen vs Adapted)"
    echo "  - Predicted segmentation masks"
    echo "  - Ground truth comparisons"
    echo "  - mIoU metrics"
    echo ""
    echo "Check SwanLab dashboard for detailed results and metrics"
    echo "========================================================================"
fi

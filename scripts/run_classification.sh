#!/bin/bash

# Classification Task Script - CIFAR-100 and ImageNet
# Supports both quick test (1 epoch) and full training
# Usage: 
#   Quick test:  bash scripts/run_classification.sh test
#   Full train:  bash scripts/run_classification.sh

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
INPUT_SIZE=224

cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "========================================================================"
echo "Classification Task - DINOv3 Multi-Task Framework"
echo "========================================================================"
echo "Mode: $MODE"
echo "Model: $DINOV3_MODEL"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"
echo ""

# =============================================================================
# Helper function
# =============================================================================

run_experiment() {
    local exp_name=$1
    local dataset=$2
    local data_path=$3
    local epochs=$4
    local batch_size=$5
    local lr=$6
    local use_daga=$7
    local daga_layers=$8
    local subset_ratio=$9
    local swanlab_name=${10}
    
    echo "▶️  Running: $exp_name"
    echo "─────────────────────────────────────────────────────────────────────"
    
    local daga_args=""
    if [ "$use_daga" = "true" ]; then
        daga_args="--use_daga --daga_layers $daga_layers"
    fi
    
    local subset_args=""
    if [ ! -z "$subset_ratio" ] && [ "$subset_ratio" != "1.0" ]; then
        subset_args="--subset_ratio $subset_ratio"
    fi
    
    python main_classification.py \
        --model_name $DINOV3_MODEL \
        --pretrained_path $PRETRAINED_PATH \
        --dataset $dataset \
        --data_path $data_path \
        --input_size $INPUT_SIZE \
        --epochs $epochs \
        --batch_size $batch_size \
        --lr $lr \
        --seed $SEED \
        $daga_args \
        $subset_args \
        --output_dir ./outputs/classification \
        --swanlab_name "$swanlab_name" \
        --enable_visualization \
        --log_freq 5 \
        --vis_indices 0 1 2 3
    
    if [ $? -eq 0 ]; then
        echo "✅  SUCCESS: $exp_name"
    else
        echo "❌  FAILED: $exp_name"
        exit 1
    fi
    echo ""
}

# =============================================================================
# Test Mode (1 epoch, 10% data)
# =============================================================================

if [ "$MODE" = "test" ]; then
    echo "🧪 QUICK TEST MODE (1 epoch, 1% data)"
    echo ""
    
    # CIFAR-100 Baseline
    run_experiment \
        "CIFAR-100 Baseline (Test)" \
        "cifar100" \
        "${DATA_ROOT}/cifar" \
        1 \
        256 \
        4e-3 \
        "false" \
        "" \
        0.01 \
        "test_cifar100_baseline"
    
    # CIFAR-100 with DAGA
    run_experiment \
        "CIFAR-100 with DAGA (Test)" \
        "cifar100" \
        "${DATA_ROOT}/cifar" \
        1 \
        256 \
        4e-3 \
        "true" \
        "11" \
        0.01 \
        "test_cifar100_daga_L11"
    
    echo "========================================================================"
    echo "✅ QUICK TEST COMPLETED!"
    echo "========================================================================"
    echo "Check outputs in: ./outputs/classification/"
    echo "========================================================================"

# =============================================================================
# Full Training Mode
# =============================================================================

else
    echo "🚀 FULL TRAINING MODE"
    echo ""
    
    # ------------------------------------------------------------------------
    # CIFAR-100 Experiments
    # ------------------------------------------------------------------------
    
    echo "========================================================================"
    echo "Part 1: CIFAR-100 Experiments"
    echo "========================================================================"
    echo ""
    
    # CIFAR-100 Baseline
    run_experiment \
        "CIFAR-100 Baseline" \
        "cifar100" \
        "${DATA_ROOT}/cifar" \
        20 \
        256 \
        4e-3 \
        "false" \
        "" \
        1.0 \
        "cifar100_baseline"
    
    # CIFAR-100 with DAGA (single layer)
    run_experiment \
        "CIFAR-100 with DAGA (L11)" \
        "cifar100" \
        "${DATA_ROOT}/cifar" \
        20 \
        256 \
        4e-3 \
        "true" \
        "11" \
        1.0 \
        "cifar100_daga_L11"
    
    # CIFAR-100 with DAGA (hourglass)
    run_experiment \
        "CIFAR-100 with DAGA (Hourglass)" \
        "cifar100" \
        "${DATA_ROOT}/cifar" \
        20 \
        256 \
        4e-3 \
        "true" \
        "1 2 10 11" \
        1.0 \
        "cifar100_daga_hourglass"
    
    # ------------------------------------------------------------------------
    # ImageNet Experiments (optional - only if ImageNet is available)
    # ------------------------------------------------------------------------
    
    if [ -d "${DATA_ROOT}/imagenet" ]; then
        echo ""
        echo "========================================================================"
        echo "Part 2: ImageNet Experiments"
        echo "========================================================================"
        echo ""
        
        # ImageNet Baseline (10% subset for faster training)
        run_experiment \
            "ImageNet Baseline (10% subset)" \
            "imagenet" \
            "${DATA_ROOT}/imagenet" \
            10 \
            128 \
            5e-5 \
            "false" \
            "" \
            0.1 \
            "imagenet_baseline_subset0.1"
        
        # ImageNet with DAGA
        run_experiment \
            "ImageNet with DAGA (10% subset)" \
            "imagenet" \
            "${DATA_ROOT}/imagenet" \
            10 \
            128 \
            5e-5 \
            "true" \
            "11" \
            0.1 \
            "imagenet_daga_L11_subset0.1"
    else
        echo "⚠️  ImageNet dataset not found at ${DATA_ROOT}/imagenet, skipping ImageNet experiments"
    fi
    
    echo ""
    echo "========================================================================"
    echo "✅ ALL CLASSIFICATION EXPERIMENTS COMPLETED!"
    echo "========================================================================"
    echo "Results saved in: ./outputs/classification/"
    echo "Check SwanLab dashboard for detailed results and visualizations"
    echo "========================================================================"
fi

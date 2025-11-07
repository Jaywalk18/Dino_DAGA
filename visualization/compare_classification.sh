#!/bin/bash
# Compare CIFAR-100 Classification Models

set -e

PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$(pwd)

source activate dinov3_env

echo "🎯 CIFAR-100 Classification Comparison"
echo "========================================"

# Find the latest model checkpoints
BASELINE_PATH=$(find outputs/classification/01_baseline -name "best_model.pth" | sort -r | head -1)
DAGA_PATH=$(find outputs/classification/02_daga_last_layer -name "best_model.pth" | sort -r | head -1)

echo "Baseline: $BASELINE_PATH"
echo "DAGA: $DAGA_PATH"

CUDA_VISIBLE_DEVICES=3,4,5,6 python visualization/compare_models.py \
    --task classification \
    --baseline_path "$BASELINE_PATH" \
    --daga_path "$DAGA_PATH" \
    --dataset cifar100 \
    --data_path /home/user/zhoutianjian/DataSets \
    --input_size 518 \
    --batch_size 32 \
    --max_samples 1000 \
    --num_visualize 10 \
    --output_dir visualization/results/classification_cifar100

echo "✅ Classification comparison complete!"


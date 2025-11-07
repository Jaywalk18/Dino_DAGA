#!/bin/bash
# Compare ADE20K Segmentation Models

set -e

PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$(pwd)

source activate dinov3_env

echo "🎨 ADE20K Segmentation Comparison"
echo "========================================"

# Find the latest model checkpoints
BASELINE_PATH=$(find outputs/segmentation/01_baseline -name "best_model.pth" | sort -r | head -1)
DAGA_PATH=$(find outputs/segmentation/02_daga_last_layer -name "best_model.pth" | sort -r | head -1)

echo "Baseline: $BASELINE_PATH"
echo "DAGA: $DAGA_PATH"

CUDA_VISIBLE_DEVICES=3,4,5,6 python visualization/compare_models.py \
    --task segmentation \
    --baseline_path "$BASELINE_PATH" \
    --daga_path "$DAGA_PATH" \
    --dataset ade20k \
    --data_path /home/user/zhoutianjian/DataSets/ADE20K_2021_17_01 \
    --input_size 518 \
    --batch_size 16 \
    --max_samples 200 \
    --num_visualize 15 \
    --output_dir visualization/results/segmentation_ade20k

echo "✅ Segmentation comparison complete!"


#!/bin/bash
# Visualization scripts for comparing baseline and DAGA models

set -e

PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
cd $PROJECT_ROOT
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate environment
source activate dinov3_env

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/visualization/results"
mkdir -p "$OUTPUT_DIR"

echo "🎨 Model Comparison Visualization Tool"
echo "========================================"

# Classification comparison
echo -e "\n📊 1. Classification Comparison"
python visualization/compare_models.py \
    --task classification \
    --baseline_path outputs/classification/01_baseline/*/best_model.pth \
    --daga_path outputs/classification/02_daga_last_layer/*/best_model.pth \
    --dataset food101 \
    --data_path /home/user/zhoutianjian/OpenDataLab___Food-101 \
    --input_size 518 \
    --batch_size 32 \
    --max_samples 1000 \
    --num_visualize 10 \
    --output_dir "$OUTPUT_DIR"

# Segmentation comparison
echo -e "\n📊 2. Segmentation Comparison"
python visualization/compare_models.py \
    --task segmentation \
    --baseline_path outputs/segmentation/01_baseline/*/best_model.pth \
    --daga_path outputs/segmentation/02_daga_last_layer/*/best_model.pth \
    --dataset ade20k \
    --data_path /home/user/zhoutianjian/DataSets/ADE20K_2021_17_01 \
    --input_size 518 \
    --batch_size 16 \
    --max_samples 500 \
    --num_visualize 10 \
    --output_dir "$OUTPUT_DIR"

echo -e "\n✅ All comparisons complete!"
echo "Results saved to: $OUTPUT_DIR"


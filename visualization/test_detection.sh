#!/bin/bash
# Test detection metrics on pretrained models

set -e

# Activate environment
source activate dinov3_env

# Project root
cd "$(dirname "$0")/.."

# Test baseline model
echo "=========================================="
echo "Testing Baseline Model"
echo "=========================================="
python visualization/test_detection_metrics.py \
    --checkpoint "outputs/detection/01_baseline/coco_baseline_L_2025-11-10/best_model.pth" \
    --data_path "/home/user/zhoutianjian/DataSets/COCO 2017" \
    --batch_size 8 \
    --sample_ratio 0.1

# Test DAGA model
echo ""
echo "=========================================="
echo "Testing DAGA Model (L1,4,7,10)"
echo "=========================================="
python visualization/test_detection_metrics.py \
    --checkpoint "outputs/detection/03_daga_detection_four_layers/coco_daga_L1-4-7-10_2025-11-10/best_model.pth" \
    --data_path "/home/user/zhoutianjian/DataSets/COCO 2017" \
    --batch_size 8 \
    --sample_ratio 0.1

echo ""
echo "✅ Testing completed! Check the detailed_metrics.txt files in each output directory."


#!/bin/bash

# Test All Tasks with Correct Data Paths
set -e

DINOV3_MODEL="dinov3_vits16"
PRETRAINED_PATH="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
INPUT_SIZE=518
TEST_EPOCHS=3
SEED=42

# Data paths
CIFAR_PATH="/home/user/zhoutianjian/DataSets/cifar"
ADE20K_PATH="/home/user/zhoutianjian/DataSets/ADE20K_2021_17_01"
COCO_PATH="/home/user/zhoutianjian/DataSets/COCO 2017"

# Ensure we're in the correct directory
if [ ! -f "main_classification.py" ]; then
    echo "Changing to Dino_DAGA directory..."
    cd /home/user/zhoutianjian/Dino_DAGA
fi

echo "========================================================================"
echo "Testing DINOv3 Multi-Task Framework"
echo "========================================================================"
echo "Python: $(which python)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# =============================================================================
# CLASSIFICATION - CIFAR-100
# =============================================================================

echo "========================================================================"
echo "[1/4] Classification - CIFAR-100 Baseline (10% data, 3 epochs)"
echo "========================================================================"

python main_classification.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset cifar100 \
    --data_path $CIFAR_PATH \
    --subset_ratio 0.1 \
    --input_size $INPUT_SIZE \
    --epochs $TEST_EPOCHS \
    --batch_size 64 \
    --lr 5e-5 \
    --seed $SEED \
    --output_dir ./test_outputs/classification \
    --swanlab_name "cifar100_baseline_test" \
    --enable_visualization \
    --log_freq 1 \
    --vis_indices 0 1 2

echo ""
echo "✓ Test 1/4 completed"
echo ""

echo "========================================================================"
echo "[2/4] Classification - CIFAR-100 with DAGA (10% data, 3 epochs)"
echo "========================================================================"

python main_classification.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset cifar100 \
    --data_path $CIFAR_PATH \
    --subset_ratio 0.1 \
    --input_size $INPUT_SIZE \
    --epochs $TEST_EPOCHS \
    --batch_size 64 \
    --lr 5e-5 \
    --seed $SEED \
    --use_daga \
    --daga_layers 11 \
    --output_dir ./test_outputs/classification \
    --swanlab_name "cifar100_daga_test" \
    --enable_visualization \
    --log_freq 1 \
    --vis_indices 0 1 2

echo ""
echo "✓ Test 2/4 completed"
echo ""

echo "========================================================================"
echo "[3/4] Segmentation - ADE20K Baseline (3 epochs)"
echo "========================================================================"

if [ -d "$ADE20K_PATH" ]; then
    python main_segmentation.py \
        --model_name $DINOV3_MODEL \
        --pretrained_path $PRETRAINED_PATH \
        --dataset ade20k \
        --data_path "$ADE20K_PATH" \
        --input_size $INPUT_SIZE \
        --epochs $TEST_EPOCHS \
        --batch_size 8 \
        --lr 1e-4 \
        --seed $SEED \
        --output_dir ./test_outputs/segmentation \
        --swanlab_name "ade20k_baseline_test" \
        --enable_visualization \
        --num_vis_samples 2 \
        --log_freq 1
    
    echo "✓ Test 3/4 completed"
else
    echo "⚠ ADE20K not found, skipped"
fi

echo ""

echo "========================================================================"
echo "[4/4] Segmentation - ADE20K with DAGA (3 epochs)"
echo "========================================================================"

if [ -d "$ADE20K_PATH" ]; then
    python main_segmentation.py \
        --model_name $DINOV3_MODEL \
        --pretrained_path $PRETRAINED_PATH \
        --dataset ade20k \
        --data_path "$ADE20K_PATH" \
        --input_size $INPUT_SIZE \
        --epochs $TEST_EPOCHS \
        --batch_size 8 \
        --lr 1e-4 \
        --seed $SEED \
        --use_daga \
        --daga_layers 11 \
        --output_dir ./test_outputs/segmentation \
        --swanlab_name "ade20k_daga_test" \
        --enable_visualization \
        --num_vis_samples 2 \
        --log_freq 1
    
    echo "✓ Test 4/4 completed"
else
    echo "⚠ ADE20K not found, skipped"
fi

echo ""
echo "========================================================================"
echo "✅ ALL TESTS COMPLETED!"
echo "========================================================================"
echo "Results in: ./test_outputs/"
echo "SwanLab: Check dashboard for visualizations"
echo "========================================================================"

#!/bin/bash

# Quick Test Script - Tests all tasks with small subsets
# This script verifies the modular framework works correctly

set -e

DINOV3_MODEL="dinov3_vits16"
PRETRAINED_PATH="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
INPUT_SIZE=518
TEST_EPOCHS=2
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
echo "QUICK TEST - DINOv3 Multi-Task Framework"
echo "========================================================================"
echo "This will run fast tests on all tasks (2 epochs, small subsets)"
echo ""

# =============================================================================
# CLASSIFICATION - CIFAR-100 (10% subset)
# =============================================================================

echo "========================================================================"
echo "[1/6] Classification - CIFAR-100 Baseline (10% subset)"
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
    --swanlab_name "test_cifar100_baseline" \
    --enable_visualization \
    --log_freq 1 \
    --vis_indices 0 1 2

echo ""
echo "✓ Test 1/6 completed"
echo ""

# =============================================================================

echo "========================================================================"
echo "[2/6] Classification - CIFAR-100 with DAGA (10% subset)"
echo "========================================================================"

python main_classification.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset cifar100 \
    --data_path ./data/cifar100 \
    --subset_ratio 0.1 \
    --input_size $INPUT_SIZE \
    --epochs $TEST_EPOCHS \
    --batch_size 64 \
    --lr 5e-5 \
    --seed $SEED \
    --use_daga \
    --daga_layers 11 \
    --output_dir ./test_outputs/classification \
    --swanlab_name "test_cifar100_daga" \
    --enable_visualization \
    --log_freq 1 \
    --vis_indices 0 1 2

echo ""
echo "✓ Test 2/6 completed"
echo ""

# =============================================================================
# SEGMENTATION - ADE20K (small subset)
# =============================================================================

echo "========================================================================"
echo "[3/6] Segmentation - ADE20K Baseline (first 100 samples)"
echo "========================================================================"

# Note: Adjust path to your ADE20K dataset
# ADE20K_PATH is already defined at the top of the script

if [ -d "$ADE20K_PATH" ]; then
    python main_segmentation.py \
        --model_name $DINOV3_MODEL \
        --pretrained_path $PRETRAINED_PATH \
        --dataset ade20k \
        --data_path $ADE20K_PATH \
        --input_size $INPUT_SIZE \
        --epochs $TEST_EPOCHS \
        --batch_size 8 \
        --lr 1e-4 \
        --seed $SEED \
        --output_dir ./test_outputs/segmentation \
        --swanlab_name "test_ade20k_baseline" \
        --enable_visualization \
        --num_vis_samples 2 \
        --log_freq 1
    
    echo ""
    echo "✓ Test 3/6 completed"
else
    echo "⚠ ADE20K dataset not found at $ADE20K_PATH, skipping..."
    echo "✗ Test 3/6 skipped"
fi

echo ""

# =============================================================================

echo "========================================================================"
echo "[4/6] Segmentation - ADE20K with DAGA (first 100 samples)"
echo "========================================================================"

if [ -d "$ADE20K_PATH" ]; then
    python main_segmentation.py \
        --model_name $DINOV3_MODEL \
        --pretrained_path $PRETRAINED_PATH \
        --dataset ade20k \
        --data_path $ADE20K_PATH \
        --input_size $INPUT_SIZE \
        --epochs $TEST_EPOCHS \
        --batch_size 8 \
        --lr 1e-4 \
        --seed $SEED \
        --use_daga \
        --daga_layers 11 \
        --output_dir ./test_outputs/segmentation \
        --swanlab_name "test_ade20k_daga" \
        --enable_visualization \
        --num_vis_samples 2 \
        --log_freq 1
    
    echo ""
    echo "✓ Test 4/6 completed"
else
    echo "⚠ ADE20K dataset not found, skipping..."
    echo "✗ Test 4/6 skipped"
fi

echo ""

# =============================================================================
# DETECTION - COCO (small subset)
# =============================================================================

echo "========================================================================"
echo "[5/6] Detection - COCO Baseline (first 100 samples)"
echo "========================================================================"

# COCO_PATH is already defined at the top of the script

if [ -d "$COCO_PATH" ]; then
    python main_detection.py \
        --model_name $DINOV3_MODEL \
        --pretrained_path $PRETRAINED_PATH \
        --dataset coco \
        --data_path $COCO_PATH \
        --input_size $INPUT_SIZE \
        --epochs $TEST_EPOCHS \
        --batch_size 8 \
        --lr 1e-4 \
        --seed $SEED \
        --output_dir ./test_outputs/detection \
        --swanlab_name "test_coco_baseline" \
        --enable_visualization \
        --num_vis_samples 2 \
        --log_freq 1
    
    echo ""
    echo "✓ Test 5/6 completed"
else
    echo "⚠ COCO dataset not found at $COCO_PATH, skipping..."
    echo "✗ Test 5/6 skipped"
fi

echo ""

# =============================================================================

echo "========================================================================"
echo "[6/6] Detection - COCO with DAGA (first 100 samples)"
echo "========================================================================"

if [ -d "$COCO_PATH" ]; then
    python main_detection.py \
        --model_name $DINOV3_MODEL \
        --pretrained_path $PRETRAINED_PATH \
        --dataset coco \
        --data_path $COCO_PATH \
        --input_size $INPUT_SIZE \
        --epochs $TEST_EPOCHS \
        --batch_size 8 \
        --lr 1e-4 \
        --seed $SEED \
        --use_daga \
        --daga_layers 11 \
        --output_dir ./test_outputs/detection \
        --swanlab_name "test_coco_daga" \
        --enable_visualization \
        --num_vis_samples 2 \
        --log_freq 1
    
    echo ""
    echo "✓ Test 6/6 completed"
else
    echo "⚠ COCO dataset not found, skipping..."
    echo "✗ Test 6/6 skipped"
fi

echo ""

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "========================================================================"
echo "QUICK TEST SUMMARY"
echo "========================================================================"
echo ""
echo "Test results saved in ./test_outputs/"
echo "  - ./test_outputs/classification/"
echo "  - ./test_outputs/segmentation/"
echo "  - ./test_outputs/detection/"
echo ""
echo "Check SwanLab dashboard for visualizations:"
echo "  - Attention map comparisons"
echo "  - Task-specific visualizations"
echo "  - Training metrics"
echo ""
echo "========================================================================"
echo "✅ QUICK TEST COMPLETED!"
echo "========================================================================"

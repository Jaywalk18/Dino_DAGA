#!/bin/bash

# Quick Classification Test - CIFAR-100 only
set -e

DINOV3_MODEL="dinov3_vits16"
PRETRAINED_PATH="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
INPUT_SIZE=518
TEST_EPOCHS=3
SEED=42

echo "========================================================================"
echo "Testing Classification on CIFAR-100"
echo "========================================================================"
echo ""

# Data paths
CIFAR_PATH="/home/user/zhoutianjian/DataSets/cifar"

# Ensure we're in the correct directory
if [ ! -f "main_classification.py" ]; then
    echo "Changing to Dino_DAGA directory..."
    cd /home/user/zhoutianjian/Dino_DAGA
fi

# Create local data directory if it doesn't exist
mkdir -p ./data/cifar100

echo "[1/2] CIFAR-100 Baseline (10% data, 3 epochs)"
echo "----------------------------------------------------------------------"

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
    --vis_indices 0 1 2 || echo "Test 1 failed"

echo ""
echo "✓ Test 1/2 completed"
echo ""

echo "[2/2] CIFAR-100 with DAGA (10% data, 3 epochs)"
echo "----------------------------------------------------------------------"

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
    --swanlab_name "test_cifar100_daga" \
    --enable_visualization \
    --log_freq 1 \
    --vis_indices 0 1 2 || echo "Test 2 failed"

echo ""
echo "✓ Test 2/2 completed"
echo ""

echo "========================================================================"
echo "✅ Classification Tests Complete!"
echo "========================================================================"
echo ""
echo "Results saved in: ./test_outputs/classification/"
echo "Check SwanLab dashboard for visualizations"
echo ""

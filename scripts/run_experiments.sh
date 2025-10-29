#!/bin/bash

# DINOv3 Multi-Task Experiments with DAGA
# This script runs classification, segmentation, and detection experiments

set -e

# Configuration
DINOV3_MODEL="dinov3_vits16"
PRETRAINED_PATH="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
INPUT_SIZE=518
EPOCHS=20
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

# =============================================================================
# CLASSIFICATION EXPERIMENTS
# =============================================================================

echo "========================================================================"
echo "CLASSIFICATION EXPERIMENTS"
echo "========================================================================"

# CIFAR-100 Baseline
echo "Running CIFAR-100 Baseline..."
python main_classification.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset cifar100 \
    --data_path $CIFAR_PATH \
    --input_size $INPUT_SIZE \
    --epochs $EPOCHS \
    --batch_size 128 \
    --lr 5e-5 \
    --seed $SEED \
    --output_dir ./outputs/classification \
    --enable_visualization \
    --log_freq 5

# CIFAR-100 with DAGA
echo "Running CIFAR-100 with DAGA..."
python main_classification.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset cifar100 \
    --data_path ./data/cifar100 \
    --input_size $INPUT_SIZE \
    --epochs $EPOCHS \
    --batch_size 128 \
    --lr 5e-5 \
    --seed $SEED \
    --use_daga \
    --daga_layers 11 \
    --output_dir ./outputs/classification \
    --enable_visualization \
    --log_freq 5

# ImageNet-100 subset Baseline
echo "Running ImageNet-100 (10% subset) Baseline..."
python main_classification.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset imagenet100 \
    --data_path $CIFAR_PATH \
    --subset_ratio 0.1 \
    --input_size $INPUT_SIZE \
    --epochs 10 \
    --batch_size 64 \
    --lr 5e-5 \
    --seed $SEED \
    --output_dir ./outputs/classification \
    --enable_visualization \
    --log_freq 2

# ImageNet-100 subset with DAGA
echo "Running ImageNet-100 (10% subset) with DAGA..."
python main_classification.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset imagenet100 \
    --data_path $CIFAR_PATH \
    --subset_ratio 0.1 \
    --input_size $INPUT_SIZE \
    --epochs 10 \
    --batch_size 64 \
    --lr 5e-5 \
    --seed $SEED \
    --use_daga \
    --daga_layers 11 \
    --output_dir ./outputs/classification \
    --enable_visualization \
    --log_freq 2

# =============================================================================
# SEGMENTATION EXPERIMENTS
# =============================================================================

echo ""
echo "========================================================================"
echo "SEGMENTATION EXPERIMENTS"
echo "========================================================================"

# ADE20K Baseline
echo "Running ADE20K Baseline..."
python main_segmentation.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset ade20k \
    --data_path $ADE20K_PATH \
    --input_size $INPUT_SIZE \
    --epochs $EPOCHS \
    --batch_size 16 \
    --lr 1e-4 \
    --seed $SEED \
    --output_dir ./outputs/segmentation \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 5

# ADE20K with DAGA
echo "Running ADE20K with DAGA..."
python main_segmentation.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset ade20k \
    --data_path $ADE20K_PATH \
    --input_size $INPUT_SIZE \
    --epochs $EPOCHS \
    --batch_size 16 \
    --lr 1e-4 \
    --seed $SEED \
    --use_daga \
    --daga_layers 11 \
    --output_dir ./outputs/segmentation \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 5

# COCO Segmentation Baseline
echo "Running COCO Segmentation Baseline..."
python main_segmentation.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset coco \
    --data_path $COCO_PATH \
    --input_size $INPUT_SIZE \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --seed $SEED \
    --output_dir ./outputs/segmentation \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 2

# COCO Segmentation with DAGA
echo "Running COCO Segmentation with DAGA..."
python main_segmentation.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset coco \
    --data_path $COCO_PATH \
    --input_size $INPUT_SIZE \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --seed $SEED \
    --use_daga \
    --daga_layers 11 \
    --output_dir ./outputs/segmentation \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 2

# =============================================================================
# DETECTION EXPERIMENTS
# =============================================================================

echo ""
echo "========================================================================"
echo "DETECTION EXPERIMENTS"
echo "========================================================================"

# COCO Detection Baseline
echo "Running COCO Detection Baseline..."
python main_detection.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset coco \
    --data_path $COCO_PATH \
    --input_size $INPUT_SIZE \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --seed $SEED \
    --output_dir ./outputs/detection \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 2

# COCO Detection with DAGA
echo "Running COCO Detection with DAGA..."
python main_detection.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset coco \
    --data_path $COCO_PATH \
    --input_size $INPUT_SIZE \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --seed $SEED \
    --use_daga \
    --daga_layers 11 \
    --output_dir ./outputs/detection \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 2

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo "Results saved in ./outputs/"
echo "  - ./outputs/classification/"
echo "  - ./outputs/segmentation/"
echo "  - ./outputs/detection/"
echo "========================================================================"

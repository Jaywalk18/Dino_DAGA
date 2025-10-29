#!/bin/bash

# Complete Experiment Suite - All Tasks with All Datasets
# Classification: CIFAR-10, CIFAR-100, ImageNet-100
# Segmentation: ADE20K, COCO
# Detection: COCO

set -e

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

echo "========================================================================"
echo "COMPLETE EXPERIMENT SUITE - DINOv3 Multi-Task with DAGA"
echo "========================================================================"
echo ""

# =============================================================================
# CLASSIFICATION EXPERIMENTS
# =============================================================================

echo "========================================================================"
echo "PART 1: CLASSIFICATION EXPERIMENTS"
echo "========================================================================"
echo ""

# --- CIFAR-10 ---
echo "[1] CIFAR-10 Baseline"
python main_classification.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset cifar10 \
    --data_path $CIFAR_PATH \
    --input_size $INPUT_SIZE \
    --epochs $EPOCHS \
    --batch_size 128 \
    --lr 5e-5 \
    --seed $SEED \
    --output_dir ./outputs/classification \
    --swanlab_name "cifar10_baseline" \
    --enable_visualization \
    --log_freq 5

echo "[2] CIFAR-10 with DAGA"
python main_classification.py \
    --model_name $DINOV3_MODEL \
    --pretrained_path $PRETRAINED_PATH \
    --dataset cifar10 \
    --data_path $CIFAR_PATH \
    --input_size $INPUT_SIZE \
    --epochs $EPOCHS \
    --batch_size 128 \
    --lr 5e-5 \
    --seed $SEED \
    --use_daga \
    --daga_layers 11 \
    --output_dir ./outputs/classification \
    --swanlab_name "cifar10_daga_L11" \
    --enable_visualization \
    --log_freq 5

# --- CIFAR-100 ---
echo "[3] CIFAR-100 Baseline"
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
    --swanlab_name "cifar100_baseline" \
    --enable_visualization \
    --log_freq 5

echo "[4] CIFAR-100 with DAGA"
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
    --use_daga \
    --daga_layers 11 \
    --output_dir ./outputs/classification \
    --swanlab_name "cifar100_daga_L11" \
    --enable_visualization \
    --log_freq 5

# --- ImageNet-100 (10% subset for faster training) ---
echo "[5] ImageNet-100 Baseline (10% subset)"
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
    --swanlab_name "imagenet100_baseline_subset0.1" \
    --enable_visualization \
    --log_freq 2

echo "[6] ImageNet-100 with DAGA (10% subset)"
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
    --swanlab_name "imagenet100_daga_L11_subset0.1" \
    --enable_visualization \
    --log_freq 2

# =============================================================================
# SEGMENTATION EXPERIMENTS
# =============================================================================

echo ""
echo "========================================================================"
echo "PART 2: SEGMENTATION EXPERIMENTS"
echo "========================================================================"
echo ""

# --- ADE20K ---
echo "[7] ADE20K Baseline"
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
    --swanlab_name "ade20k_baseline" \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 5

echo "[8] ADE20K with DAGA"
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
    --swanlab_name "ade20k_daga_L11" \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 5

# --- COCO Segmentation ---
echo "[9] COCO Segmentation Baseline"
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
    --swanlab_name "coco_seg_baseline" \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 2

echo "[10] COCO Segmentation with DAGA"
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
    --swanlab_name "coco_seg_daga_L11" \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 2

# =============================================================================
# DETECTION EXPERIMENTS
# =============================================================================

echo ""
echo "========================================================================"
echo "PART 3: DETECTION EXPERIMENTS"
echo "========================================================================"
echo ""

# --- COCO Detection ---
echo "[11] COCO Detection Baseline"
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
    --swanlab_name "coco_det_baseline" \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 2

echo "[12] COCO Detection with DAGA"
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
    --swanlab_name "coco_det_daga_L11" \
    --enable_visualization \
    --num_vis_samples 4 \
    --log_freq 2

# =============================================================================
# COMPLETION
# =============================================================================

echo ""
echo "========================================================================"
echo "✅ ALL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo ""
echo "Results saved in ./outputs/"
echo "  Classification:  ./outputs/classification/"
echo "  Segmentation:    ./outputs/segmentation/"
echo "  Detection:       ./outputs/detection/"
echo ""
echo "Experiment Summary:"
echo "  Classification: 6 experiments (CIFAR-10, CIFAR-100, ImageNet-100)"
echo "  Segmentation:   4 experiments (ADE20K, COCO)"
echo "  Detection:      2 experiments (COCO)"
echo "  Total:         12 experiments"
echo ""
echo "Check SwanLab dashboard for detailed results and visualizations!"
echo "========================================================================"

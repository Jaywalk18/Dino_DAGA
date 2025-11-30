#!/bin/bash
# Classification training script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"
# ============================================================================
# Common Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/classification"
INPUT_SIZE=224
BATCH_SIZE=256
NUM_WORKERS=16
SAMPLE_RATIO=""  # e.g. "0.1" for 10%
LOG_FREQ=5

# ============================================================================
# Dataset Configuration (uncomment one block to use)
# ============================================================================

# --- CIFAR-100 (100 classes, 50K/10K, 32x32) ---
DATASET="cifar100"
DATA_PATH="/home/user/zhoutianjian/DataSets/cifar"
EPOCHS=20
LR=2e-1

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Classification"

run_experiment "main_classification.py" "01_baseline" "Baseline"

run_experiment "main_classification.py" "04_daga_hourglass_layer" "DAGA (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11

echo -e "\n🎉 Classification training completed!"

# --- Food-101 (101 classes, 75K/25K) ---
# DATASET="food101"
# DATA_PATH="/home/user/zhoutianjian/DataSets/food-101"
# EPOCHS=20
# LR=5e-3

# --- ImageNet (1000 classes, 1.2M/50K) ---
# DATASET="imagenet"
# DATA_PATH="/home/user/zhoutianjian/DataSets/imagenet"
# EPOCHS=50
# LR=5e-1



# --- Oxford Pets (37 classes, 3.7K/3.7K) ---
DATASET="pets"
DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___Oxford-IIIT_Pets/raw"
EPOCHS=100
LR=5e-3


setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Classification"

run_experiment "main_classification.py" "01_baseline" "Baseline"

run_experiment "main_classification.py" "04_daga_hourglass_layer" "DAGA (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11

echo -e "\n🎉 Classification training completed!"


# --- Stanford Cars (196 classes, 8K/8K) ---
DATASET="cars"
DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___Stanford_Cars/raw/Stanford_Cars"
EPOCHS=60
LR=5e-3

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Classification"

run_experiment "main_classification.py" "01_baseline" "Baseline"

run_experiment "main_classification.py" "04_daga_hourglass_layer" "DAGA (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11

echo -e "\n🎉 Classification training completed!"


# --- SUN397 (397 classes, ~80K/20K) ---
DATASET="sun397"
DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___SUN397/raw/SUN397"
EPOCHS=30
LR=5e-3


setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Classification"

run_experiment "main_classification.py" "01_baseline" "Baseline"

run_experiment "main_classification.py" "04_daga_hourglass_layer" "DAGA (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11

echo -e "\n🎉 Classification training completed!"


# --- DTD (47 classes, ~2K/2K) ---
DATASET="dtd"
DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___DTD/raw/dtd"
EPOCHS=100
LR=5e-3

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Classification"

run_experiment "main_classification.py" "01_baseline" "Baseline"

run_experiment "main_classification.py" "04_daga_hourglass_layer" "DAGA (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11

echo -e "\n🎉 Classification training completed!"



# --- Flowers-102 (102 classes, 1K train/6K test) ---
# NOTE: Small dataset, use smaller batch size to avoid empty dataloader in DDP
DATASET="flowers102"
DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___Oxford_102_Flower/raw"
EPOCHS=100
LR=5e-3
BATCH_SIZE=32  

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

print_config "Classification"

run_experiment "main_classification.py" "01_baseline" "Baseline"

run_experiment "main_classification.py" "04_daga_hourglass_layer" "DAGA (L1,L2,L10,L11)" \
    --use_daga --daga_layers 1 2 10 11

echo -e "\n🎉 Classification training completed!"
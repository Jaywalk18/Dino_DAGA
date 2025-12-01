#!/bin/bash
# Classification training script
# Tests different DAGA insertion layer configurations
# LR values optimized based on grid search results
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Common Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/classification"
INPUT_SIZE=224
NUM_WORKERS=16
SAMPLE_RATIO=""
LOG_FREQ=5

# ============================================================================
# DAGA Layer Configurations (from Table 5)
# ============================================================================
# Last only {11}        -> 88.8%
# Middle {5, 8, 11}     -> 89.6%
# Four layers {2,5,8,11}-> 91.0%
# Hourglass {1,2,10,11} -> 91.9%

run_all_daga_configs() {
    # 01. Baseline (no DAGA)
    run_experiment "main_classification.py" "01_baseline" "Baseline"
    
    # 02. Last only {11}
    run_experiment "main_classification.py" "02_daga_last_only" "DAGA (L11)" \
        --use_daga --daga_layers 11
    
    # 03. Middle {5, 8, 11}
    run_experiment "main_classification.py" "03_daga_middle" "DAGA (L5,L8,L11)" \
        --use_daga --daga_layers 5 8 11
    
    # 04. Four layers {2, 5, 8, 11}
    run_experiment "main_classification.py" "04_daga_four_layers" "DAGA (L2,L5,L8,L11)" \
        --use_daga --daga_layers 2 5 8 11
    
    # 05. Hourglass {1, 2, 10, 11} - Best config
    run_experiment "main_classification.py" "05_daga_hourglass" "DAGA (L1,L2,L10,L11)" \
        --use_daga --daga_layers 1 2 10 11
}

# ============================================================================
# 1. CIFAR-10 (10 classes, 50K/10K) - Best LR: 0.5-1.0
# ============================================================================
DATASET="cifar10"
DATA_PATH="/home/user/zhoutianjian/DataSets/cifar"
EPOCHS=30
LR=0.5
BATCH_SIZE=256

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Classification - CIFAR-10"
run_all_daga_configs
echo -e "\n✅ CIFAR-10 completed!"

# ============================================================================
# 2. CIFAR-100 (100 classes, 50K/10K) - Best LR: 0.5
# ============================================================================
DATASET="cifar100"
DATA_PATH="/home/user/zhoutianjian/DataSets/cifar"
EPOCHS=30
LR=0.5
BATCH_SIZE=256

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Classification - CIFAR-100"
run_all_daga_configs
echo -e "\n✅ CIFAR-100 completed!"

# ============================================================================
# 3. Flowers-102 (102 classes, 1K/6K) - Best LR: 0.1
# ============================================================================
DATASET="flowers102"
DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___Oxford_102_Flower/raw"
EPOCHS=50
LR=0.1
BATCH_SIZE=32

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Classification - Flowers-102"
run_all_daga_configs
echo -e "\n✅ Flowers-102 completed!"

# ============================================================================
# 4. DTD (47 classes, ~2K/2K) - Best LR: 0.5
# ============================================================================
DATASET="dtd"
DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___DTD/raw/dtd"
EPOCHS=50
LR=0.5
BATCH_SIZE=64

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Classification - DTD"
run_all_daga_configs
echo -e "\n✅ DTD completed!"

# ============================================================================
# 5. Oxford Pets (37 classes, 3.7K/3.7K) - Estimated LR: 0.1
# ============================================================================
DATASET="pets"
DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___Oxford-IIIT_Pets/raw"
EPOCHS=50
LR=0.1
BATCH_SIZE=128

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Classification - Oxford Pets"
run_all_daga_configs
echo -e "\n✅ Oxford Pets completed!"

# ============================================================================
# 6. Stanford Cars (196 classes, 8K/8K) - Estimated LR: 0.2
# ============================================================================
DATASET="cars"
DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___Stanford_Cars/raw/Stanford_Cars"
EPOCHS=50
LR=0.2
BATCH_SIZE=128

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Classification - Stanford Cars"
run_all_daga_configs
echo -e "\n✅ Stanford Cars completed!"

# ============================================================================
# 7. Food-101 (101 classes, 75K/25K) - Estimated LR: 0.2
# ============================================================================
DATASET="food101"
DATA_PATH="/home/user/zhoutianjian/DataSets/food-101"
EPOCHS=20
LR=0.2
BATCH_SIZE=256

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Classification - Food-101"
run_all_daga_configs
echo -e "\n✅ Food-101 completed!"

# ============================================================================
# 8. SUN397 (397 classes, ~80K/20K) - Estimated LR: 0.2
# ============================================================================
DATASET="sun397"
DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___SUN397/raw/SUN397"
EPOCHS=30
LR=0.2
BATCH_SIZE=256

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Classification - SUN397"
run_all_daga_configs
echo -e "\n✅ SUN397 completed!"

# ============================================================================
# 9. ImageNet (1000 classes, 1.2M/50K) - Estimated LR: 0.3
# ============================================================================
DATASET="imagenet"
DATA_PATH="/home/user/zhoutianjian/DataSets/imagenet"
EPOCHS=30
LR=0.3
BATCH_SIZE=256

setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"
print_config "Classification - ImageNet"
run_all_daga_configs
echo -e "\n✅ ImageNet completed!"

# ============================================================================
echo -e "\n🎉 All classification experiments completed!"
echo "Datasets: CIFAR-10, CIFAR-100, Flowers-102, DTD, Pets, Cars, Food-101, SUN397, ImageNet"
echo "DAGA configs: Baseline, Last{11}, Middle{5,8,11}, Four{2,5,8,11}, Hourglass{1,2,10,11}"

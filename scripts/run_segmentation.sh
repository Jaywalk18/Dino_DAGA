#!/bin/bash
# Semantic Segmentation training script
# Supports ADE20K, VOC2012, and Cityscapes datasets
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Common Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/segmentation"
NUM_WORKERS=8
LOG_FREQ=5
NUM_VIS_SAMPLES=4
OUT_INDICES="2 5 8 11"  # Multi-layer features for segmentation

# ============================================================================
# DAGA Configurations
# ============================================================================
run_all_daga_configs() {
    # 01. Baseline (no DAGA)
    run_experiment "main_segmentation.py" "01_baseline" "Baseline"
    
    # 02. Last only {11}
    run_experiment "main_segmentation.py" "02_daga_last_only" "DAGA (L11)" \
        --use_daga --daga_layers 11
    
    # 03. Middle {5, 8, 11}
    run_experiment "main_segmentation.py" "03_daga_middle" "DAGA (L5,L8,L11)" \
        --use_daga --daga_layers 5 8 11
    
    # 04. Four layers {2, 5, 8, 11}
    run_experiment "main_segmentation.py" "04_daga_four_layers" "DAGA (L2,L5,L8,L11)" \
        --use_daga --daga_layers 2 5 8 11
    
    # 05. Hourglass {1, 2, 10, 11}
    run_experiment "main_segmentation.py" "05_daga_hourglass" "DAGA (L1,L2,L10,L11)" \
        --use_daga --daga_layers 1 2 10 11
}

# ============================================================================
# 1. ADE20K (150 classes, 20K train, 2K val) - Original dataset
# ============================================================================
run_ade20k() {
    echo ""
    echo "=========================================="
    echo "🎨 ADE20K Segmentation"
    echo "=========================================="
    
    DATASET="ade20k"
    DATA_PATH="/home/user/zhoutianjian/DataSets/ADE20K_2021_17_01"
    INPUT_SIZE=518
    EPOCHS=50
    LR=5e-3
    BATCH_SIZE=16
    
    setup_environment
    setup_paths
    mkdir -p "$BASE_OUTPUT_DIR"
    print_config "Segmentation - ADE20K"
    run_all_daga_configs
    echo -e "\n✅ ADE20K completed!"
}

# ============================================================================
# 2. PASCAL VOC 2012 (21 classes, ~2.9K train, ~1.5K val)
# ============================================================================
run_voc2012() {
    echo ""
    echo "=========================================="
    echo "📦 PASCAL VOC 2012 Segmentation"
    echo "=========================================="
    
    DATASET="voc2012"
    DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___PASCAL_VOC2012/raw/VOCdevkit/VOC2012"
    INPUT_SIZE=512
    EPOCHS=50
    LR=1e-4
    BATCH_SIZE=8
    
    setup_environment
    setup_paths
    mkdir -p "$BASE_OUTPUT_DIR"
    print_config "Segmentation - VOC2012"
    run_all_daga_configs
    echo -e "\n✅ VOC2012 completed!"
}

# ============================================================================
# 2. Cityscapes (19 classes, 2975 train, 500 val)
# ============================================================================
run_cityscapes() {
    echo ""
    echo "=========================================="
    echo "🏙️ Cityscapes Segmentation"
    echo "=========================================="
    
    DATASET="cityscapes"
    DATA_PATH="/home/user/zhoutianjian/DataSets/OpenDataLab___CityScapes/raw"
    INPUT_SIZE=512
    EPOCHS=80
    LR=1e-4
    BATCH_SIZE=4  # Cityscapes images are larger
    
    setup_environment
    setup_paths
    mkdir -p "$BASE_OUTPUT_DIR"
    print_config "Segmentation - Cityscapes"
    run_all_daga_configs
    echo -e "\n✅ Cityscapes completed!"
}

# ============================================================================
# Main
# ============================================================================
if [ $# -ge 1 ]; then
    case $1 in
        ade20k|ade)
            run_ade20k
            ;;
        voc2012|voc)
            run_voc2012
            ;;
        cityscapes|city)
            run_cityscapes
            ;;
        all)
            run_ade20k
            run_voc2012
            run_cityscapes
            echo -e "\n🎉 All segmentation experiments completed!"
            ;;
        *)
            echo "Unknown dataset: $1"
            echo "Usage: ./run_segmentation.sh [ade20k|voc2012|cityscapes|all]"
            exit 1
            ;;
    esac
else
    echo "Usage: ./run_segmentation.sh [ade20k|voc2012|cityscapes|all]"
    echo ""
    echo "Available datasets:"
    echo "  ade20k     - ADE20K (150 classes) - Original"
    echo "  voc2012    - PASCAL VOC 2012 (21 classes)"
    echo "  cityscapes - Cityscapes (19 classes)"
    echo "  all        - Run all datasets"
fi

#!/bin/bash
# Learning Rate Grid Search for Classification
# Quick search using 3 epochs per LR to find optimal range
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# LR Search Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/lr_search"
INPUT_SIZE=224
NUM_WORKERS=16
SAMPLE_RATIO=""
LOG_FREQ=10
SEARCH_EPOCHS=3  # Quick search with few epochs

# LR candidates (log scale from 0.001 to 1.0)
LR_CANDIDATES=(0.001 0.005 0.01 0.05 0.1 0.2 0.5 1.0)

# ============================================================================
# LR Search Function
# ============================================================================
run_lr_search() {
    local dataset=$1
    local data_path=$2
    local batch_size=$3
    
    echo ""
    echo "=============================================="
    echo "🔍 LR Search: $dataset"
    echo "=============================================="
    echo "LR candidates: ${LR_CANDIDATES[*]}"
    echo "Epochs per LR: $SEARCH_EPOCHS"
    echo ""
    
    RESULTS_FILE="${BASE_OUTPUT_DIR}/${dataset}_lr_results.txt"
    mkdir -p "$BASE_OUTPUT_DIR"
    echo "# LR Search Results for $dataset" > "$RESULTS_FILE"
    echo "# Format: LR -> Best Val Acc" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    BEST_LR=""
    BEST_ACC="0"
    
    # Set dataset variables for run_experiment
    DATASET="$dataset"
    DATA_PATH="$data_path"
    BATCH_SIZE="$batch_size"
    EPOCHS="$SEARCH_EPOCHS"
    
    for lr in "${LR_CANDIDATES[@]}"; do
        echo ""
        echo ">>> Testing LR=$lr"
        
        LR="$lr"
        OUTPUT_SUBDIR="${BASE_OUTPUT_DIR}/${dataset}/lr_${lr}"
        
        # Run experiment with DAGA Hourglass config {1,2,10,11}
        run_experiment "main_classification.py" "lr_${lr}" "LR=$lr (DAGA Hourglass)" \
            --use_daga --daga_layers 1 2 10 11 2>&1 | tee "${OUTPUT_SUBDIR}/train.log" || true
        
        # Extract best accuracy from log
        ACC=$(grep -oP 'val_acc: \K[0-9.]+' "${OUTPUT_SUBDIR}/train.log" 2>/dev/null | tail -1 || echo "0")
        
        echo "LR=$lr -> Val Acc=$ACC" | tee -a "$RESULTS_FILE"
        
        # Track best using awk (bc not available)
        if awk "BEGIN {exit !($ACC > $BEST_ACC)}"; then
            BEST_ACC="$ACC"
            BEST_LR="$lr"
        fi
    done
    
    echo "" | tee -a "$RESULTS_FILE"
    echo "=============================================="
    echo "✅ Best LR for $dataset: $BEST_LR (Acc: $BEST_ACC)"
    echo "=============================================="
    echo "BEST: LR=$BEST_LR -> Acc=$BEST_ACC" >> "$RESULTS_FILE"
}

# ============================================================================
# Main
# ============================================================================
setup_environment
setup_paths
mkdir -p "$BASE_OUTPUT_DIR"

if [ $# -ge 1 ]; then
    TARGET=$1
    
    case $TARGET in
        cifar10)
            run_lr_search "cifar10" "/home/user/zhoutianjian/DataSets/cifar" 256
            ;;
        cifar100)
            run_lr_search "cifar100" "/home/user/zhoutianjian/DataSets/cifar" 256
            ;;
        food101)
            run_lr_search "food101" "/home/user/zhoutianjian/DataSets/food-101" 256
            ;;
        pets)
            run_lr_search "pets" "/home/user/zhoutianjian/DataSets/OpenDataLab___Oxford-IIIT_Pets/raw" 128
            ;;
        cars)
            run_lr_search "cars" "/home/user/zhoutianjian/DataSets/OpenDataLab___Stanford_Cars/raw/Stanford_Cars" 128
            ;;
        sun397)
            run_lr_search "sun397" "/home/user/zhoutianjian/DataSets/OpenDataLab___SUN397/raw/SUN397" 256
            ;;
        dtd)
            run_lr_search "dtd" "/home/user/zhoutianjian/DataSets/OpenDataLab___DTD/raw/dtd" 64
            ;;
        flowers102)
            run_lr_search "flowers102" "/home/user/zhoutianjian/DataSets/OpenDataLab___Oxford_102_Flower/raw" 32
            ;;
        imagenet)
            run_lr_search "imagenet" "/home/user/zhoutianjian/DataSets/imagenet" 256
            ;;
        all)
            echo "🔍 Running LR search for ALL datasets..."
            # Small datasets first (fast)
            run_lr_search "cifar10" "/home/user/zhoutianjian/DataSets/cifar" 256
            run_lr_search "cifar100" "/home/user/zhoutianjian/DataSets/cifar" 256
            run_lr_search "flowers102" "/home/user/zhoutianjian/DataSets/OpenDataLab___Oxford_102_Flower/raw" 32
            run_lr_search "dtd" "/home/user/zhoutianjian/DataSets/OpenDataLab___DTD/raw/dtd" 64
            run_lr_search "pets" "/home/user/zhoutianjian/DataSets/OpenDataLab___Oxford-IIIT_Pets/raw" 128
            run_lr_search "cars" "/home/user/zhoutianjian/DataSets/OpenDataLab___Stanford_Cars/raw/Stanford_Cars" 128
            # Medium datasets
            run_lr_search "food101" "/home/user/zhoutianjian/DataSets/food-101" 256
            run_lr_search "sun397" "/home/user/zhoutianjian/DataSets/OpenDataLab___SUN397/raw/SUN397" 256
            # Large dataset last
            run_lr_search "imagenet" "/home/user/zhoutianjian/DataSets/imagenet" 256
            echo ""
            echo "🎉 LR search completed for all datasets!"
            echo "Results saved in: $BASE_OUTPUT_DIR/"
            ;;
        *)
            echo "Unknown dataset: $TARGET"
            echo "Usage: ./lr_search.sh <dataset>"
            echo "Available: cifar10, cifar100, food101, pets, cars, sun397, dtd, flowers102, imagenet, all"
            exit 1
            ;;
    esac
else
    echo "Usage: ./lr_search.sh <dataset>"
    echo ""
    echo "Available datasets:"
    echo "  cifar10, cifar100, food101, pets, cars, sun397, dtd, flowers102, imagenet"
    echo "  all - Run search for all datasets"
    echo ""
    echo "LR candidates: ${LR_CANDIDATES[*]}"
    echo "Search epochs: $SEARCH_EPOCHS"
fi

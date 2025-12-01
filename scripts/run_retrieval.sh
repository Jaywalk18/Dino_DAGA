#!/bin/bash
# Instance Retrieval evaluation script
# Supports ROxford5k and RParis6k datasets
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# ============================================================================
# Common Settings
# ============================================================================
BASE_OUTPUT_DIR="outputs/retrieval"
INPUT_SIZE=224
NUM_WORKERS=8
POOLING="gem"  # Options: cls, avg, gem

# ============================================================================
# DAGA Configurations
# ============================================================================
run_all_daga_configs() {
    # 01. Baseline (no DAGA)
    run_experiment "main_retrieval.py" "01_baseline" "Baseline"
    
    # 02. Last only {11}
    run_experiment "main_retrieval.py" "02_daga_last_only" "DAGA (L11)" \
        --use_daga --daga_layers 11
    
    # 03. Hourglass {1, 2, 10, 11}
    run_experiment "main_retrieval.py" "03_daga_hourglass" "DAGA (L1,L2,L10,L11)" \
        --use_daga --daga_layers 1 2 10 11
}

# ============================================================================
# 1. Revisited Oxford (ROxford5k)
# ============================================================================
run_roxford() {
    echo ""
    echo "=========================================="
    echo "🏛️ Revisited Oxford5k (ROxford)"
    echo "=========================================="
    
    DATASET="roxford5k"
    DATA_PATH="/home/user/zhoutianjian/DataSets/Oxford5k"
    EPOCHS=0  # No training for baseline evaluation
    LR=1e-4
    BATCH_SIZE=32
    
    setup_environment
    setup_paths
    mkdir -p "$BASE_OUTPUT_DIR"
    print_config "Retrieval - ROxford5k"
    run_all_daga_configs
    echo -e "\n✅ ROxford5k completed!"
}

# ============================================================================
# 2. Revisited Paris (RParis6k)
# ============================================================================
run_rparis() {
    echo ""
    echo "=========================================="
    echo "🗼 Revisited Paris6k (RParis)"
    echo "=========================================="
    
    DATASET="rparis6k"
    DATA_PATH="/home/user/zhoutianjian/DataSets/Paris6k"
    EPOCHS=0  # No training for baseline evaluation
    LR=1e-4
    BATCH_SIZE=32
    
    setup_environment
    setup_paths
    mkdir -p "$BASE_OUTPUT_DIR"
    print_config "Retrieval - RParis6k"
    run_all_daga_configs
    echo -e "\n✅ RParis6k completed!"
}

# ============================================================================
# Main
# ============================================================================
if [ $# -ge 1 ]; then
    case $1 in
        roxford|oxford)
            run_roxford
            ;;
        rparis|paris)
            run_rparis
            ;;
        all)
            run_roxford
            run_rparis
            echo -e "\n🎉 All retrieval experiments completed!"
            ;;
        *)
            echo "Unknown dataset: $1"
            echo "Usage: ./run_retrieval.sh [roxford|rparis|all]"
            exit 1
            ;;
    esac
else
    echo "Usage: ./run_retrieval.sh [roxford|rparis|all]"
    echo ""
    echo "Available datasets:"
    echo "  roxford - Revisited Oxford5k (5063 images, 70 queries)"
    echo "  rparis  - Revisited Paris6k (6412 images, 70 queries)"
    echo "  all     - Run both datasets"
    echo ""
    echo "Evaluation protocols: Easy, Medium, Hard"
    echo "Metrics: mAP, R@1, R@5, R@10, R@100"
fi


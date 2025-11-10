#!/bin/bash
# Master script to run all evaluation tasks
# This script runs all 6 evaluation types: 3 main tasks + 3 additional evals

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "DINOv3 + DAGA Complete Evaluation Suite"
echo "=========================================="
echo ""
echo "This script will run all 9 evaluation types:"
echo "  Main Tasks:"
echo "    1. Classification"
echo "    2. Detection"
echo "    3. Segmentation"
echo ""
echo "  Additional Evaluations:"
echo "    4. KNN (K-Nearest Neighbors)"
echo "    5. Linear Probe"
echo "    6. Logistic Regression"
echo "    7. Depth Estimation"
echo "    8. Robustness (ImageNet-C)"
echo "    9. Text-Image Alignment (DINOtxt)"
echo ""
echo "=========================================="
echo ""

# Function to run a task with error handling
run_task() {
    local task_name=$1
    local script_path=$2
    
    echo ""
    echo "=========================================="
    echo "Running: $task_name"
    echo "=========================================="
    
    if bash "$script_path"; then
        echo "✅ $task_name completed successfully"
    else
        echo "❌ $task_name failed"
        return 1
    fi
}

# Optional: Run main tasks (already tested and working)
# Uncomment the lines below if you want to run the main tasks as well

# echo ""
# echo "================================================"
# echo "PART 1: Main Tasks (Already Tested)"
# echo "================================================"
# 
# run_task "Classification" "${SCRIPT_DIR}/run_classification.sh"
# run_task "Detection" "${SCRIPT_DIR}/run_detection.sh"
# run_task "Segmentation" "${SCRIPT_DIR}/run_segmentation.sh"

echo ""
echo "================================================"
echo "PART 2: Additional Evaluations (New)"
echo "================================================"

run_task "KNN Evaluation" "${SCRIPT_DIR}/run_knn.sh"
run_task "Linear Probe Evaluation" "${SCRIPT_DIR}/run_linear.sh"
run_task "Logistic Regression Evaluation" "${SCRIPT_DIR}/run_logreg.sh"
run_task "Depth Estimation" "${SCRIPT_DIR}/run_depth.sh"
run_task "Robustness Evaluation (ImageNet-C)" "${SCRIPT_DIR}/run_robustness.sh"
run_task "Text-Image Alignment (DINOtxt)" "${SCRIPT_DIR}/run_dinotxt.sh"

echo ""
echo "=========================================="
echo "🎉 All evaluations completed!"
echo "=========================================="
echo ""
echo "Results can be found in:"
echo "  - outputs/knn/"
echo "  - outputs/linear/"
echo "  - outputs/logreg/"
echo "  - outputs/depth/"
echo "  - outputs/robustness/"
echo "  - outputs/dinotxt/"
echo ""


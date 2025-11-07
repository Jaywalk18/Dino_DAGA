#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
export OMP_NUM_THREADS=1
# --- Configuration ---
DATASET="imagenet100"
DATA_PATH="../../DataSets/ImageNet100"
BASE_OUTPUT_DIR="experiments_ablation"
GPU_IDS=${1:-"1,2,3,4,5,6"}

# --- Define the two experiments to compare ---
BASELINE_EXP_NAME="01_baseline"
PLUGIN_EXP_NAME="02_daga_only_fewer_layer"

# --- Path Logic ---
BASELINE_MODEL_PATH="${BASE_OUTPUT_DIR}/${BASELINE_EXP_NAME}/${DATASET}_${BASELINE_EXP_NAME}/best_model.pth"
PLUGIN_MODEL_PATH="${BASE_OUTPUT_DIR}/${PLUGIN_EXP_NAME}/${DATASET}_${PLUGIN_EXP_NAME}/best_model.pth"

# --- Parameters ---
NUM_TOP_DIFF=30
COMPARISON_OUTPUT_DIR="comparison_results/${BASELINE_EXP_NAME}_vs_${PLUGIN_EXP_NAME}_on_${DATASET}"

# --- ✨ MODIFICATION: Calculate number of GPUs for torchrun ---
# Count the number of commas and add 1 to get the GPU count.
NUM_GPUS=$(echo "$GPU_IDS" | awk -F, '{print NF}')

# --- Setup ---
mkdir -p "$COMPARISON_OUTPUT_DIR"
echo "🚀 Starting Model Comparison on ${NUM_GPUS} GPUs..."
echo "==================================================================="
echo "  GPU ID(s):            ${GPU_IDS}"
echo "  Dataset:              ${DATASET}"
echo "-------------------------------------------------------------------"
echo "  Baseline Model Path:  ${BASELINE_MODEL_PATH}"
echo "  Plugin Model Path:    ${PLUGIN_MODEL_PATH}"
echo "-------------------------------------------------------------------"
echo "  Output Directory:     ${COMPARISON_OUTPUT_DIR}"
echo "==================================================================="

# --- Check if model files exist ---
if [ ! -f "$BASELINE_MODEL_PATH" ]; then
    echo "❌ ERROR: Baseline model not found at ${BASELINE_MODEL_PATH}"
    exit 1
fi
if [ ! -f "$PLUGIN_MODEL_PATH" ]; then
    echo "❌ ERROR: Plugin model not found at ${PLUGIN_MODEL_PATH}"
    exit 1
fi

# --- ✨ MODIFICATION: Run the script using torchrun for multi-GPU execution ---
echo -e "\n▶️  Running comparison analysis..."

# torchrun will handle setting up the environment for each process.
# Each process will get a unique 'local_rank' from 0 to NUM_GPUS-1.
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS compare_models.py \
    --baseline_model_path "$BASELINE_MODEL_PATH" \
    --plugin_model_path "$PLUGIN_MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --dataset "$DATASET" \
    --num_top_diff "$NUM_TOP_DIFF" \
    --output_dir "$COMPARISON_OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✅  SUCCESS: Comparison complete. Visualizations saved to '${COMPARISON_OUTPUT_DIR}'"
else
    echo "❌  FAILED: Comparison script encountered an error."
    exit 1
fi

echo -e "\n🎉 Analysis finished!"

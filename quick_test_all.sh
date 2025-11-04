#!/bin/bash
# 快速测试所有三个任务（1 epoch，10-15%数据）
set -e

PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
cd $PROJECT_ROOT

GPU_IDS=${1:-"1,2,3,4,5,6"}

echo "========================================================================"
echo "          DINOv3 DAGA 快速测试（所有任务）"
echo "========================================================================"
echo "GPU IDs: ${GPU_IDS}"
echo "Project: ${PROJECT_ROOT}"
echo ""
echo "此脚本将依次运行："
echo "  1. 分类任务 (CIFAR-100, 15%数据, 1 epoch)"
echo "  2. 检测任务 (COCO, 12000样本, 1 epoch)"
echo "  3. 分割任务 (ADE20K, 2000样本, 1 epoch)"
echo "========================================================================"
echo ""

# 询问用户是否继续
read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)

# 1. 分类任务
echo ""
echo "========================================================================"
echo "[1/3] 运行分类任务..."
echo "========================================================================"
./scripts/run_classification.sh ${GPU_IDS}

# 2. 检测任务
echo ""
echo "========================================================================"
echo "[2/3] 运行检测任务..."
echo "========================================================================"
./scripts/run_detection.sh ${GPU_IDS}

# 3. 分割任务
echo ""
echo "========================================================================"
echo "[3/3] 运行分割任务..."
echo "========================================================================"
./scripts/run_segmentation.sh ${GPU_IDS}

# 计算总时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "========================================================================"
echo "✅ 所有任务完成！"
echo "========================================================================"
echo "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
echo ""
echo "结果保存在:"
echo "  - outputs/classification/"
echo "  - outputs/detection/"
echo "  - outputs/segmentation/"
echo "========================================================================"


#!/usr/bin/env python3
"""
验证DINOv3 DAGA项目设置是否正确
"""
import os
import sys

def check_file_exists(path, name):
    """检查文件是否存在"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {name}: {path}")
    return exists

def check_directory_exists(path, name):
    """检查目录是否存在"""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    print(f"{status} {name}: {path}")
    return exists

def main():
    print("=" * 70)
    print("DINOv3 DAGA 项目设置验证")
    print("=" * 70)
    
    all_good = True
    
    # 检查数据集路径
    print("\n[数据集检查]")
    datasets = [
        ("/home/user/zhoutianjian/DataSets/cifar", "CIFAR-100"),
        ("/home/user/zhoutianjian/DataSets/COCO 2017", "COCO 2017"),
        ("/home/user/zhoutianjian/DataSets/ADE20K_2021_17_01", "ADE20K"),
    ]
    for path, name in datasets:
        if not check_directory_exists(path, name):
            all_good = False
    
    # 检查模型权重
    print("\n[模型权重检查]")
    checkpoint_dir = "/home/user/zhoutianjian/DAGA/checkpoints"
    if check_directory_exists(checkpoint_dir, "Checkpoint目录"):
        checkpoint_file = os.path.join(checkpoint_dir, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
        if not check_file_exists(checkpoint_file, "DINOv3-ViT-S/16权重"):
            all_good = False
    else:
        all_good = False
    
    # 检查核心模块
    print("\n[核心模块检查]")
    core_files = [
        "core/backbones.py",
        "core/daga.py",
        "core/heads.py",
        "core/utils.py",
    ]
    for file in core_files:
        if not check_file_exists(file, os.path.basename(file)):
            all_good = False
    
    # 检查任务模块
    print("\n[任务模块检查]")
    task_files = [
        "tasks/classification.py",
        "tasks/detection.py",
        "tasks/segmentation.py",
    ]
    for file in task_files:
        if not check_file_exists(file, os.path.basename(file)):
            all_good = False
    
    # 检查数据加载模块
    print("\n[数据加载模块检查]")
    data_files = [
        "data/classification_datasets.py",
        "data/detection_datasets.py",
        "data/segmentation_datasets.py",
    ]
    for file in data_files:
        if not check_file_exists(file, os.path.basename(file)):
            all_good = False
    
    # 检查主程序
    print("\n[主程序检查]")
    main_files = [
        "main_classification.py",
        "main_detection.py",
        "main_segmentation.py",
    ]
    for file in main_files:
        if not check_file_exists(file, os.path.basename(file)):
            all_good = False
    
    # 检查训练脚本
    print("\n[训练脚本检查]")
    script_files = [
        "scripts/run_classification.sh",
        "scripts/run_detection.sh",
        "scripts/run_segmentation.sh",
    ]
    for file in script_files:
        if not check_file_exists(file, os.path.basename(file)):
            all_good = False
    
    # 检查Python环境
    print("\n[Python环境检查]")
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
    except ImportError:
        print("✗ PyTorch未安装")
        all_good = False
    
    try:
        import torchvision
        print(f"✓ torchvision: {torchvision.__version__}")
    except ImportError:
        print("✗ torchvision未安装")
        all_good = False
    
    # 最终结果
    print("\n" + "=" * 70)
    if all_good:
        print("✓ 所有检查通过！可以开始训练。")
        print("\n运行示例：")
        print("  source activate dinov3_env")
        print("  cd /home/user/zhoutianjian/Dino_DAGA")
        print("  ./scripts/run_classification.sh")
    else:
        print("✗ 部分检查失败，请检查上述错误。")
    print("=" * 70)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())


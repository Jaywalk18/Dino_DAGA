# DINOv3 with DAGA Plugin

此项目实现了DINOv3与DAGA（Dynamic Attention-Guided Adaptation）插件的集成，支持分类、检测和分割三个下游任务。

## 环境设置

```bash
source activate dinov3_env
```

## 项目结构

```
Dino_DAGA/
├── core/                    # 核心模块
│   ├── backbones.py        # DINOv3骨干网络加载
│   ├── daga.py             # DAGA模块实现
│   ├── heads.py            # 任务头（分类、检测、分割）
│   └── utils.py            # 工具函数
├── data/                    # 数据集加载
│   ├── classification_datasets.py
│   ├── detection_datasets.py
│   └── segmentation_datasets.py
├── tasks/                   # 任务实现
│   ├── classification.py   # 分类任务
│   ├── detection.py        # 检测任务
│   └── segmentation.py     # 分割任务
├── scripts/                 # 训练脚本
│   ├── run_classification.sh
│   ├── run_detection.sh
│   └── run_segmentation.sh
├── main_classification.py  # 分类主程序
├── main_detection.py       # 检测主程序
└── main_segmentation.py    # 分割主程序
```

## 数据集

- **分类**: CIFAR-100, ImageNet (位于 `/home/user/zhoutianjian/DataSets/`)
- **检测**: COCO 2017 (位于 `/home/user/zhoutianjian/DataSets/COCO 2017`)
- **分割**: ADE20K (位于 `/home/user/zhoutianjian/DataSets/ADE20K_2021_17_01`)

## 模型权重

DINOv3预训练权重位于 `/home/user/zhoutianjian/DAGA/checkpoints/`

## 使用方法

### 1. 分类任务

```bash
cd /home/user/zhoutianjian/Dino_DAGA
./scripts/run_classification.sh
```

配置说明：
- Dataset: CIFAR-100
- Training Subset: 15%
- Batch Size: 256
- Learning Rate: 4e-3
- Epochs: 1

### 2. 检测任务

```bash
cd /home/user/zhoutianjian/Dino_DAGA
./scripts/run_detection.sh
```

配置说明：
- Dataset: COCO 2017
- Training Samples: 12000 (~10%)
- Batch Size: 16
- Learning Rate: 1e-4
- Epochs: 1

输出指标包括：
- mAP (Mean Average Precision)
- Precision
- Recall
- F1 Score

### 3. 分割任务

```bash
cd /home/user/zhoutianjian/Dino_DAGA
./scripts/run_segmentation.sh
```

配置说明：
- Dataset: ADE20K
- Training Samples: 2000 (~10%)
- Batch Size: 16
- Learning Rate: 1e-4
- Epochs: 1

输出指标包括：
- mIoU (Mean Intersection over Union)
- Pixel Accuracy

## GPU配置

默认使用GPU 1-6，可通过参数修改：

```bash
./scripts/run_classification.sh 0,1,2,3  # 使用GPU 0-3
```

## 输出结果

训练结果保存在 `outputs/` 目录下：
- `outputs/classification/` - 分类结果和可视化
- `outputs/detection/` - 检测结果和可视化
- `outputs/segmentation/` - 分割结果和可视化

## 关键改进

### 分类任务
- 使用 **CLS token + patch tokens mean** 作为特征（与DINOv3官方一致）
- 改进的线性分类头初始化

### 检测任务
- 完整的mAP计算
- 改进的检测头架构
- 优化的学习率策略

### 分割任务
- 多尺度特征融合
- 改进的分割头（包含BatchNorm和Dropout）
- Label smoothing正则化

## DAGA插件

DAGA模块动态适配特征：
1. **Attention Encoder**: 从注意力图提取指导信息
2. **Gate Generator**: 生成动态门控
3. **Feature Transformer**: 特征变换
4. **Adaptive Mixing**: 自适应混合原始和变换后的特征

## 注意事项

- 测试使用10-15%数据集以快速验证
- 完整训练需要增加epochs和数据比例
- 建议batch size根据GPU内存调整
- DINOv3性能较好，如果结果很差请检查数据路径和模型加载

## 依赖

- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm
- swanlab (用于实验记录)
- pycocotools (用于COCO数据集)

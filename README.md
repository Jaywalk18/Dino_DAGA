# DINOv3 + DAGA

This repository implements DAGA (Dynamic Attention-Guided Adaptation) on top of DINOv3 for three vision tasks:
- **Classification** (ImageNet-1K)
- **Object Detection** (COCO)
- **Semantic Segmentation** (ADE20K)

## Quick Start

### 1. Environment Setup
```bash
source activate dinov3_env
cd /home/user/zhoutianjian/Dino_DAGA
```

### 2. Run Training

**Classification:**
```bash
cd scripts
./run_classification.sh [GPU_IDS]
```

**Detection:**
```bash
cd scripts
./run_detection.sh [GPU_IDS]
```

**Segmentation:**
```bash
cd scripts
./run_segmentation.sh [GPU_IDS]
```

Default GPU_IDS: `1,2,3,4,5,6`

## Project Structure

```
Dino_DAGA/
├── core/
│   ├── backbones.py      # DINOv3 backbone loading
│   ├── daga.py           # DAGA module implementation
│   ├── heads.py          # Task-specific heads
│   └── utils.py          # Utilities
├── data/
│   ├── detection_datasets.py
│   └── segmentation_datasets.py
├── tasks/
│   ├── classification.py
│   ├── detection.py
│   └── segmentation.py
├── scripts/
│   ├── run_classification.sh
│   ├── run_detection.sh
│   └── run_segmentation.sh
└── main_*.py             # Entry points
```

## Key Features

### Multi-layer Feature Extraction
Following official DINOv3 practices:
- Detection & Segmentation: Use layers [2, 5, 8, 11]
- Classification: Use final layer + global pooling

### Improved Loss Functions
- **Detection**: Focal Loss + GIoU Loss
- **Segmentation**: CrossEntropy with label smoothing
- **Classification**: Standard CrossEntropy

### Training Configuration
Aligned with official DINOv3 recommendations:
- **Detection**: 24 epochs, full COCO dataset
- **Segmentation**: 40 epochs (~40K iterations), full ADE20K
- **Classification**: 50 epochs

## DAGA Module

DAGA adapts frozen backbone features using attention-guided dynamic gating:
```python
# Enable DAGA on layer 11
--use_daga --daga_layers 11
```

## Notes

- All models freeze the DINOv3 backbone (only train heads + DAGA)
- Lightweight segmentation head following official design
- Multi-scale features for dense prediction tasks
- Automatic visualization during training

# DAGA: Dynamic Attention-Guided Adaptation for Vision Foundation Models

Official implementation of DAGA method based on DINOv3.

---

## Overview

DAGA (Dynamic Attention-Guided Adaptation) is a parameter-efficient fine-tuning method for vision foundation models. The key idea is to use frozen backbone attention maps as spatial guidance signals for dynamic feature adaptation.

**Key Components:**
- **Attention Encoder**: Encodes attention maps into compact instruction vectors
- **Dynamic Gate Generator**: Produces instance-specific gating signals
- **Feature Transformer**: Applies gated transformations with residual connections

**Supported Tasks:**
- Image Classification (CIFAR-10/100, ImageNet)
- Object Detection (COCO)
- Semantic Segmentation (ADE20K)
- Monocular Depth Estimation (NYU Depth v2)
- Robustness Evaluation (ImageNet-C/A/R)
- Linear Probing & k-NN Evaluation

---

## Installation

### Environment Setup

**⚠️ Important: Always activate the virtual environment before running any scripts!**

```bash
# Create conda environment with Python 3.11
conda create -n dinov3_env python=3.11 -y
conda activate dinov3_env

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DINOv3
cd dinov3
pip install -e .
cd ..

# Install additional dependencies
pip install -r requirements.txt
```

### Dependencies List

The project requires the following packages (included in `requirements.txt`):

**Core Dependencies:**
- `torch` >= 2.0.0 (with CUDA support)
- `torchvision` >= 0.15.0
- `tqdm` >= 4.65.0
- `numpy` >= 1.23.0

**Logging & Visualization:**
- `swanlab` >= 0.7.0 (experiment tracking)
- `matplotlib` >= 3.7.0
- `opencv-python` >= 4.8.0

**Data Processing:**
- `pandas` >= 2.0.0
- `openpyxl` >= 3.1.0
- `h5py` (for NYU Depth V2)

**Task-Specific:**
- `pycocotools` >= 2.0.6 (for COCO detection)

### Verify Installation

```bash
conda activate dinov3_env
python -c "import dinov3; print('✓ DINOv3 installed')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import swanlab; print('✓ SwanLab installed')"
python -c "import pycocotools; print('✓ pycocotools installed')"
```

---

## Quick Start

### 0. Pre-flight Checklist ✅

Before running any experiments, make sure:

```bash
# 1. Activate environment
conda activate dinov3_env

# 2. Verify environment
which python  # Should show: /home/user/.conda/envs/dinov3_env/bin/python

# 3. Test imports
python << EOF
import torch
import dinov3
import swanlab
import pandas
import pycocotools
import h5py
print("✅ All dependencies installed!")
EOF

# 4. Check GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

### 1. Download Pretrained Weights

```bash
mkdir -p /path/to/checkpoints
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain_lvd1689m.pth \
    -O /path/to/checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth
```

### 2. Configure Paths

Edit `scripts/common_config.sh`:

```bash
# Path settings
PROJECT_ROOT="/home/user/zhoutianjian/Dino_DAGA"
CHECKPOINT_DIR="/path/to/checkpoints"

# GPU Configuration
DEFAULT_GPU_IDS="0,1,2,3"
```

### 3. Run Training

We recommend using the shell scripts in `scripts/` directory:

```bash
# Image Classification
bash scripts/run_classification.sh

# Object Detection
bash scripts/run_detection.sh

# Semantic Segmentation
bash scripts/run_segmentation.sh

# Depth Estimation
bash scripts/run_depth.sh
```

### 4. Evaluation Tasks

```bash
# k-NN evaluation
bash scripts/run_knn.sh

# Linear probing
bash scripts/run_linear.sh

# Logistic regression
bash scripts/run_logreg.sh

# Robustness evaluation
bash scripts/run_robustness.sh
```

---

## Training Configuration

### Task-Specific Scripts

Each script in `scripts/` contains task-specific configurations. You can modify:

- `DATASET`: Dataset to use
- `DATA_PATH`: Path to dataset
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size per GPU
- `LR`: Learning rate
- `INPUT_SIZE`: Input image resolution
- `NUM_WORKERS`: Number of data loading workers

### DAGA Configuration

Apply DAGA to specific transformer layers:

```bash
# Hourglass configuration (recommended)
--use_daga --daga_layers 1 2 10 11

# Single layer
--use_daga --daga_layers 11

# Multiple layers
--use_daga --daga_layers 0 2 5 8 11
```

### SwanLab Logging

SwanLab is used for experiment tracking with workspace: `NUDT_SSL__CVPR`

**Project mapping:**
- Classification → `DINOv3-ImageNet-Classification`
- Detection → `DINOv3-COCO-Detection`
- Segmentation → `DINOv3-ADE20K-Segmentation`
- Depth → `DINOv3-NYUv2-Depth`
- Robustness → `DINOv3-ImageNet-C-Robustness`
- Linear/LogReg/KNN → `DINOv3-Linear-Probing` / `DINOv3-Logistic-Regression` / `DINOv3-KNN-Evaluation`

**About X-axis (step vs epoch):**
- **Training tasks** (Classification, Detection, Segmentation, Depth): Use `epoch` as step
- **Evaluation tasks** (Linear, LogReg, KNN): Use `iteration` as step (these don't have traditional epochs)

To disable SwanLab:

```bash
# In scripts/common_config.sh
export SWANLAB_MODE=disabled
```

---

## Direct Python Usage

If you prefer running Python scripts directly:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    main_classification.py \
    --dataset imagenet \
    --data_path /path/to/imagenet \
    --model_name dinov3_vitb16 \
    --pretrained_path /path/to/checkpoint.pth \
    --use_daga \
    --daga_layers 1 2 10 11 \
    --batch_size 128 \
    --epochs 20 \
    --lr 2e-2
```

---

## Project Structure

```
Dino_DAGA/
├── core/                      # Core modules
│   ├── daga.py               # DAGA implementation
│   ├── backbones.py          # Backbone loaders
│   ├── heads.py              # Task-specific heads
│   └── utils.py              # Utilities
├── data/                      # Dataset loaders
├── tasks/                     # Task implementations
├── scripts/                   # Training scripts (recommended)
│   ├── common_config.sh      # Shared configuration
│   ├── run_classification.sh
│   ├── run_detection.sh
│   └── ...
├── dinov3/                    # DINOv3 submodule
├── main_*.py                  # Entry points
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### DataLoader Shared Memory Error

If you encounter:
```
ERROR: Unexpected bus error encountered in worker.
This might be caused by insufficient shared memory (shm).
```

**Check shared memory:**
```bash
df -h /dev/shm
```

**Solution 1: Increase shared memory (requires root)**
```bash
sudo mount -o remount,size=16G /dev/shm
```

**Solution 2: Reduce workers (no root required)**
```bash
# In scripts/*.sh
NUM_WORKERS=0
```

### CUDA Out of Memory

- Reduce `BATCH_SIZE`
- Reduce `INPUT_SIZE`
- Use fewer GPUs

### Module Not Found

**⚠️ IMPORTANT: Always activate dinov3_env before running any scripts!**

```bash
# Check current environment
conda info --envs

# Activate the correct environment
conda activate dinov3_env

# Verify you're in the right environment
which python
# Should output: /home/user/.conda/envs/dinov3_env/bin/python

# Install missing dependencies
pip install -r requirements.txt
cd dinov3 && pip install -e . && cd ..
```

**Common missing modules:**
- `pandas` → `pip install pandas`
- `openpyxl` → `pip install openpyxl`
- `pycocotools` → `pip install pycocotools`
- `h5py` → `pip install h5py`

**Quick fix for all dependencies:**
```bash
conda activate dinov3_env
pip install pandas openpyxl pycocotools h5py
```

---

## License

This project is released under the MIT License. DINOv3 components follow their original license.

---

## Acknowledgments

This work builds upon:
- [DINOv3](https://github.com/facebookresearch/dinov3) - Self-supervised vision transformer
- [ViT-Adapter](https://github.com/czczup/ViT-Adapter) - Vision Transformer Adapter


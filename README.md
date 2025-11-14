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

```bash
# Create conda environment
conda create -n dinov3_env python=3.11 -y
conda activate dinov3_env

# Install DINOv3
cd dinov3
pip install -e .

# Install additional dependencies
cd ..
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import dinov3; print('✓ DINOv3 installed')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
```

---

## Quick Start

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

SwanLab is used for experiment tracking. To disable:

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

```bash
conda activate dinov3_env
pip install -r requirements.txt
cd dinov3 && pip install -e .
```

---

## License

This project is released under the MIT License. DINOv3 components follow their original license.

---

## Acknowledgments

This work builds upon:
- [DINOv3](https://github.com/facebookresearch/dinov3) - Self-supervised vision transformer
- [ViT-Adapter](https://github.com/czczup/ViT-Adapter) - Vision Transformer Adapter


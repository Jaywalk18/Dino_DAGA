# Dino_DAGA

Dynamic Attention-Gated Adapter (DAGA) for DINOv3 Multi-Task Fine-tuning

## Overview

This project implements a multi-task fine-tuning framework based on DINOv3 vision transformer, enhanced with Dynamic Attention-Gated Adapter (DAGA) for improved performance across multiple computer vision tasks.

### Supported Tasks

- **Image Classification**: CIFAR-100, ImageNet
- **Object Detection**: COCO
- **Semantic Segmentation**: ADE20K

### Key Features

- ✓ Modular Design - Shared backbone, task-specific heads
- ✓ SwanLab Integration - Automatic metric & visualization logging
- ✓ Attention Visualization - Frozen backbone vs adapted model comparison
- ✓ Task-Specific Visualizations:
  - Classification: Attention maps + predictions
  - Segmentation: GT/pred masks + attention maps
  - Detection: Bounding boxes + attention maps

## Quick Start

### 1. Environment Setup

```bash
# Activate conda environment
source activate dinov3_env

# Set up project paths
export PYTHONPATH=$PYTHONPATH:/home/user/zhoutianjian/Dino_DAGA
```

### 2. Quick Test (1 epoch)

Test each task with a quick run:

```bash
# Classification
bash scripts/run_classification.sh test

# Detection
bash scripts/run_detection.sh test

# Segmentation
bash scripts/run_segmentation.sh test
```

### 3. Full Training

Run complete experiments:

```bash
# Classification (CIFAR-100 and ImageNet)
bash scripts/run_classification.sh

# Detection (COCO)
bash scripts/run_detection.sh

# Segmentation (ADE20K)
bash scripts/run_segmentation.sh
```

## Project Structure

```
Dino_DAGA/
├── core/                    # Core components
│   ├── backbones.py        # DINOv3 backbone loader
│   ├── daga.py             # DAGA implementation
│   ├── heads.py            # Task-specific heads
│   └── utils.py            # Utilities and logging
├── tasks/                   # Task implementations
│   ├── classification.py   # Classification pipeline
│   ├── detection.py        # Detection pipeline
│   └── segmentation.py     # Segmentation pipeline
├── data/                    # Data loading
│   ├── classification_datasets.py
│   ├── detection_datasets.py
│   └── segmentation_datasets.py
├── scripts/                 # Training scripts
│   ├── run_classification.sh
│   ├── run_detection.sh
│   └── run_segmentation.sh
├── main_classification.py   # Classification entry
├── main_detection.py        # Detection entry
└── main_segmentation.py     # Segmentation entry
```

## DAGA Architecture

The Dynamic Attention-Gated Adapter consists of four core components:

1. **AttentionEncoder**: Extracts instruction vectors from attention maps
2. **DynamicGateGenerator**: Generates dynamic gating signals
3. **FeatureTransformer**: Transforms and enhances features
4. **DAGA Integration**: Combines all components for adaptive feature modification

## Configuration

### Paths

- **Datasets**: `/home/user/zhoutianjian/DataSets/`
  - CIFAR: `cifar/`
  - ImageNet: `imagenet/`
  - COCO: `COCO 2017/`
  - ADE20K: `ADE20K_2021_17_01/`
- **Checkpoints**: `/home/user/zhoutianjian/DAGA/checkpoints/`

### Environment

- **Conda Environment**: `dinov3_env`
- **GPUs**: 1,2,3,4,5,6 (6 GPUs)

## Results

Training outputs are saved in `./outputs/<task>/`, including:
- Model checkpoints
- Training logs
- Visualization results

SwanLab dashboard provides:
- Training/validation metrics
- Learning rate curves
- Attention map comparisons
- Task-specific visualizations

## Advanced Usage

### Command-Line Options

Common parameters for all tasks:

```bash
--model_name dinov3_vits16      # DINOv3 model variant
--pretrained_path <path>         # Path to pretrained weights
--epochs 20                      # Training epochs
--batch_size 128                 # Batch size
--lr 5e-5                        # Learning rate
--seed 42                        # Random seed
--use_daga                       # Enable DAGA
--daga_layers 11                 # DAGA insertion layers
--enable_swanlab                 # Enable SwanLab logging
--enable_visualization           # Enable attention visualization
```

Task-specific parameters:

**Classification:**
```bash
--dataset cifar100               # Dataset choice
--subset_ratio 0.1               # Use subset of data
--vis_indices 0 1 2 3            # Visualization sample indices
```

**Detection/Segmentation:**
```bash
--num_vis_samples 4              # Number of visualization samples
```

### DAGA Layer Configurations

- **Single Layer**: `--daga_layers 11` (last layer only)
- **Hourglass**: `--daga_layers 1 2 10 11` (shallow + deep layers)
- **Balanced**: `--daga_layers 3 7 11` (evenly distributed)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 64`
   - Reduce input size: `--input_size 224`

2. **Dataset Not Found**
   - Check data paths in scripts
   - Ensure datasets are downloaded to correct locations

3. **Poor Performance**
   - Verify DINOv3 checkpoint is loaded correctly
   - Check data preprocessing and augmentation
   - Compare with reference implementation in `raw_code/`

## Citation

If you use this code, please cite:

```bibtex
@misc{dino_daga_2024,
  title={Dynamic Attention-Gated Adapter for DINOv3 Multi-Task Fine-tuning},
  author={Your Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- DINOv3 for the pretrained vision transformer models
- SwanLab for experiment tracking and visualization

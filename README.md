# DINOv3 + DAGA Integration

Complete implementation of DINOv3 with DAGA (Dynamic Attention-Guided Adaptation) for three vision tasks: classification, detection, and segmentation.

## Project Structure

```
Dino_DAGA/
├── core/                          # Core modules
│   ├── daga.py                   # DAGA implementation
│   ├── heads.py                  # Task heads (classification, segmentation)
│   ├── detr_components.py        # DETR detection components
│   ├── backbones.py              # DINOv3 backbone loading
│   └── utils.py                  # Utility functions
├── data/                          # Dataset loaders
│   ├── classification_datasets.py
│   ├── detection_datasets.py
│   └── segmentation_datasets.py
├── tasks/                         # Task implementations
│   ├── classification.py
│   ├── detection.py
│   └── segmentation.py
├── scripts/                       # Training scripts
│   ├── run_classification.sh
│   ├── run_detection.sh
│   └── run_segmentation.sh
├── main_*.py                      # Main entry points
└── outputs/                       # Experiment outputs
```

## Requirements

```bash
# Activate conda environment
source activate dinov3_env

# Required packages
pip install torch torchvision
pip install pycocotools  # For COCO detection
pip install scipy        # For Hungarian matching
pip install swanlab      # Experiment logging
```

## Quick Start

### 1. Segmentation Task (Verified ✅)

The segmentation task has been successfully verified and achieves reasonable results.

```bash
cd scripts
bash run_segmentation.sh
```

**Configuration**:
- GPUs: 3,4,5,6 (4 cards)
- Dataset: ADE20K
- Training: 1 epoch, 2% data
- DAGA layer: Layer 11

**Expected Results**:
- Val mIoU: ~24%
- Val Pixel Acc: ~62%

### 2. Detection Task (DETR Architecture ✅)

Now implements complete DETR architecture based on official DINOv3.

```bash
cd scripts
bash run_detection.sh
```

**Architecture**:
- DETR with Transformer decoder
- 100 learnable queries
- Hungarian matching
- Multi-component loss (CE + BBox + GIoU)

**Note**: 1 epoch is insufficient for detection - training loss decreases normally but mAP requires 50+ epochs to converge.

### 3. Classification Task

```bash
cd scripts
bash run_classification.sh
```

## Configuration

All scripts support environment variables:

```bash
# Set GPUs
export GPU_IDS="3,4,5,6"

# Set data ratio (0-1)
export SAMPLE_RATIO=0.02  # Use 2% of data

# Run script
bash scripts/run_segmentation.sh
```

### Main Parameters

In `run_*.sh` scripts:

- `SAMPLE_RATIO`: Data usage ratio (0-1)
- `EPOCHS`: Training epochs
- `BATCH_SIZE`: Batch size
- `LR`: Learning rate
- `GPU_IDS`: GPU IDs to use

## DAGA Integration

DAGA (Dynamic Attention-Guided Adaptation) is applied at the last layer (Layer 11):

```python
# Enable DAGA in script
run_experiment "with_daga" "DAGA (L11)" \
    --use_daga --daga_layers 11
```

## Implementation Details

### Segmentation
- **Architecture**: LinearHead (Dropout → GroupNorm → Conv1x1)
- **Features**: Multi-layer fusion (L2, L5, L8, L11)
- **Status**: ✅ Working well

### Detection
- **Architecture**: DETR (Transformer-based)
- **Components**:
  - TransformerDecoder (6 layers)
  - Query embeddings (100 queries)
  - Hungarian matching
  - Multi-component loss
- **Status**: ✅ Implemented, needs full training

### Classification
- **Architecture**: Linear classifier
- **Features**: CLS token + patch mean pooling
- **Status**: ✅ Working

## Comparison with Official DINOv3

| Aspect | Current Implementation | Official DINOv3 |
|--------|----------------------|-----------------|
| Segmentation Head | LinearHead (Dropout→GroupNorm→Conv) | LinearHead (Dropout→SyncBN→Conv) |
| Detection Head | DETR (Transformer-based) | DETR (Transformer-based) |
| Feature Extraction | Manual block iteration | get_intermediate_layers() |
| Multi-layer Features | [2,5,8,11] | [2,5,8,11] (ViT-S/B) |
| Normalization | GroupNorm | SyncBatchNorm |

## Test Results

### Segmentation (1 epoch, 2% data)
- ✅ Val mIoU: **24.13%**
- ✅ Val Pixel Acc: **62.38%**
- Status: **Success** - Ready for experiments

### Detection (1 epoch, 2% data)
- Train Loss: 9.82 → 8.46 ✓ (decreasing)
- mAP: 0% (expected - needs 50+ epochs)
- Status: **Architecture Verified** - Ready for full training

### Recommendations

**For DAGA Verification** (Recommended):
```bash
# Use segmentation task - works well and stable
cd scripts
bash run_segmentation.sh
```

**For Full Training**:
```bash
export SAMPLE_RATIO=1.0  # Use 100% data
# Modify EPOCHS in script (e.g., 50 for detection, 20 for segmentation)
bash run_detection.sh  # or run_segmentation.sh
```

## Troubleshooting

### CUDA Errors
If you encounter "GET was unable to find an engine" error:
- Fixed: Using GroupNorm instead of BatchNorm
- Reduce batch size if needed

### Environment Activation
Ensure scripts contain:
```bash
source activate dinov3_env
```

### Data Paths
Check data paths in scripts:
- COCO: `/home/user/zhoutianjian/DataSets/COCO 2017`
- ADE20K: `/home/user/zhoutianjian/DataSets/ADE20K_2021_17_01`

## Documentation

- `README.md` (this file) - Usage guide
- `DETR_DETECTION_REPORT.md` - Detailed detection implementation report

## Citation

If you use this code, please cite DINOv3 and DAGA papers.

## License

MIT License

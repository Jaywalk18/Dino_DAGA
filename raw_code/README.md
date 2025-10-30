# Raw Code Reference

This directory contains reference implementations used as the baseline for the project refactoring.

## Contents

- `daga.py`: Reference implementation of DAGA module (Dynamic Attention-Gated Adapter)
  - This is the canonical implementation that `core/daga.py` is based on
  - Contains all four core components: AttentionEncoder, DynamicGateGenerator, FeatureTransformer, and DAGA
  
- `run_dinov3_experiments_pth.sh`: Reference training script for classification tasks
  - Used as the baseline for creating `scripts/run_classification.sh`
  - Contains hyperparameter configurations and training procedures

## Purpose

These files are kept for reference purposes to:
1. Ensure consistency with the original successful implementation
2. Provide a baseline for troubleshooting and verification
3. Document the design decisions and parameter choices

## Note

Do not modify these files. They serve as the authoritative reference.
For actual development, use the files in:
- `core/` for model components
- `scripts/` for training scripts
- `tasks/` for task-specific implementations

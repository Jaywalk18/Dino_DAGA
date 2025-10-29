================================================================================
                    DINOv3 Multi-Task Framework with DAGA
================================================================================

Modular framework for fine-tuning DINOv3 on classification, segmentation, 
and detection tasks with Dynamic Attention-Gated Adapter (DAGA).

================================================================================
                              STRUCTURE
================================================================================

Dino_DAGA/
├── core/                    - Shared components (backbone, DAGA, heads, utils)
├── data/                    - Dataset loaders for all tasks
├── tasks/                   - Task-specific training/eval/visualization
├── main_classification.py   - Classification entry point
├── main_segmentation.py     - Segmentation entry point
├── main_detection.py        - Detection entry point
└── scripts/
    ├── quick_test.sh            - Quick test (2 epochs, small subsets)
    ├── run_all_experiments.sh   - Full experiments (all tasks, all datasets)
    └── run_experiments.sh       - Original experiment runner

================================================================================
                              QUICK START
================================================================================

1. Quick Test (recommended first):
   bash scripts/quick_test.sh

2. Single Task:
   # Classification
   python main_classification.py --dataset cifar100 --data_path ./data/cifar100 --use_daga

   # Segmentation
   python main_segmentation.py --dataset ade20k --data_path ./data/ADE20K --use_daga

   # Detection
   python main_detection.py --dataset coco --data_path ./data/coco2017 --use_daga

3. Full Experiments:
   bash scripts/run_all_experiments.sh

================================================================================
                              FEATURES
================================================================================

✓ Modular Design        - Shared backbone, task-specific heads
✓ SwanLab Integration   - Automatic metric & visualization logging
✓ Attention Viz         - Frozen backbone vs adapted model comparison
✓ Task Visualizations:
  - Classification:     Attention maps + predictions
  - Segmentation:       GT/pred masks + attention maps
  - Detection:          Bounding boxes + attention maps
✓ Multi-GPU Support     - Automatic DataParallel
✓ Flexible DAGA         - Apply to any layer combination

================================================================================
                              DATASETS
================================================================================

Classification:
  - CIFAR-10, CIFAR-100    (auto-download)
  - ImageNet-100, ImageNet (manual download required)

Segmentation:
  - ADE20K                 (manual download required)
  - COCO                   (manual download required)

Detection:
  - COCO                   (manual download required)

================================================================================
                              KEY ARGUMENTS
================================================================================

Common:
  --use_daga              Enable DAGA
  --daga_layers 11        Which layers (e.g., 11 or 8 9 10 11)
  --enable_visualization  Enable periodic visualizations
  --log_freq 5            Visualization frequency (epochs)
  --epochs 20             Number of epochs
  --batch_size 128        Batch size
  --lr 5e-5               Learning rate
  --seed 42               Random seed

Classification-specific:
  --dataset cifar10|cifar100|imagenet100|imagenet
  --subset_ratio 0.1      Use subset of data
  --vis_indices 0 1 2 3   Which images to visualize

Segmentation-specific:
  --dataset ade20k|coco
  --num_vis_samples 4     How many samples to visualize

Detection-specific:
  --dataset coco
  --num_vis_samples 4     How many samples to visualize

================================================================================
                              SWANLAB LOGGING
================================================================================

All experiments automatically log to SwanLab:
  - Training/validation metrics
  - Learning rate schedules
  - Attention map comparisons
  - Task-specific visualizations

Check the SwanLab dashboard during/after training.

================================================================================
                              EXAMPLES
================================================================================

1. Quick classification test (10% data, 2 epochs):
   python main_classification.py \
       --dataset cifar100 \
       --data_path ./data/cifar100 \
       --subset_ratio 0.1 \
       --epochs 2 \
       --batch_size 64 \
       --enable_visualization

2. Full classification with DAGA:
   python main_classification.py \
       --dataset cifar100 \
       --data_path ./data/cifar100 \
       --use_daga \
       --daga_layers 11 \
       --epochs 20 \
       --enable_visualization

3. Segmentation with multi-layer DAGA:
   python main_segmentation.py \
       --dataset ade20k \
       --data_path ./data/ADE20K \
       --use_daga \
       --daga_layers 8 9 10 11 \
       --epochs 20 \
       --enable_visualization

================================================================================
                              TROUBLESHOOTING
================================================================================

Import errors:
  cd /home/user/zhoutianjian/Dino_DAGA
  export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA out of memory:
  - Reduce --batch_size
  - Reduce --input_size

Quick testing:
  - Use --subset_ratio 0.1 for 10% of data
  - Use --epochs 2 for fast iteration
  - Run scripts/quick_test.sh

================================================================================

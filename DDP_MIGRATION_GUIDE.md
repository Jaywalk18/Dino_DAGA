# DDP Migration Summary

## ✅ COMPLETED (Segmentation)
- main_segmentation.py: Fully migrated to DDP
- tasks/segmentation.py: Added rank/world_size parameters  
- run_segmentation.sh: Using torchrun
- core/ddp_utils.py: Created shared utilities

## 📝 KEY CHANGES NEEDED FOR DETECTION & CLASSIFICATION

### 1. Main File Pattern (apply to main_detection.py & main_classification.py)

**Import Changes:**
```python
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from core.ddp_utils import setup_ddp, cleanup_ddp, create_ddp_dataloaders
```

**Main Function Pattern:**
```python
def main():
    # Setup DDP
    local_rank, rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    
    args = parse_arguments()
    setup_environment(args.seed + rank)  # Different seed per process
    
    # Only main process does logging/output setup
    if is_main_process:
        experiment_name = setup_logging(args, task_name="detection")
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Broadcast output_dir to all processes
    output_dir_list = [str(output_dir)] if is_main_process else [None]
    dist.broadcast_object_list(output_dir_list, src=0)
    if not is_main_process:
        output_dir = Path(output_dir_list[0])
    
    device = torch.device(f"cuda:{local_rank}")
    
    # ... load datasets ...
    
    # Use DDP dataloaders
    train_loader, val_loader = create_ddp_dataloaders(
        train_dataset, val_dataset, args.batch_size, world_size, rank,
        collate_fn=detection_collate_fn  # For detection only
    )
    
    # ... load model ...
    model.to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    # ... training ...
    try:
        best_metric, final_metric, total_time = run_training_loop(
            model, train_loader, val_loader, optimizer, scheduler, device, args,
            output_dir, fixed_vis_data, num_classes,
            rank=rank, world_size=world_size  # Add these!
        )
        if is_main_process:
            finalize_experiment(...)
    finally:
        cleanup_ddp()
```

### 2. Training Loop Pattern (apply to tasks/detection.py & tasks/classification.py)

**Function Signature:**
```python
def run_training_loop(
    model, train_loader, val_loader, optimizer, scheduler, device, args,
    output_dir, fixed_vis_data, num_classes,
    rank=0, world_size=1  # Add these with defaults
):
    is_main_process = (rank == 0)
    
    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_metric = train_epoch(...)
        val_metric = evaluate(...)
        
        # Only print/log on main process
        if is_main_process:
            print(f"Epoch {epoch+1}: ...")
            # visualization, checkpointing, logging
```

### 3. Shell Script Pattern (apply to run_detection.sh & run_classification.sh)

**Replace:**
```bash
CUDA_VISIBLE_DEVICES=$GPU_IDS python main_detection.py \
    --args...
```

**With:**
```bash
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    main_detection.py \
    --args...
```

## 🎯 BATCH SIZE RECOMMENDATIONS (with DDP)

Current (DataParallel, unbalanced):
- Segmentation: 16 (main GPU: 22GB, others: 2.5GB)
- Detection: 64 (would be unbalanced)
- Classification: 64 (would be unbalanced)

**New (DDP, balanced):**
- Segmentation: **32** (4 GPUs × 8/GPU = ~10-12GB each) ✅
- Detection: **128** (4 GPUs × 32/GPU = ~12-15GB each) ✅  
- Classification: **128** (4 GPUs × 32/GPU = ~8-10GB each) ✅

## 📊 EXPECTED RESULTS

### Memory Distribution
**Before (DataParallel):**
```
GPU 3: ████████████████████████ 22GB (main)
GPU 4: ████ 2.5GB
GPU 5: ████ 2.5GB
GPU 6: ████ 2.5GB
```

**After (DDP):**
```
GPU 3: ████████████ 12GB (balanced!)
GPU 4: ████████████ 12GB
GPU 5: ████████████ 12GB
GPU 6: ████████████ 12GB
```

### Training Speed
- 2-3x larger effective batch size
- ~20-30% faster per epoch (better communication)
- More stable training (no bottleneck on main GPU)

## 🚀 QUICK START

1. Already working: `bash scripts/run_segmentation.sh`
2. Need completion: Detection & Classification

Would you like me to complete the migration for detection and classification now?


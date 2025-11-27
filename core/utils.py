import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
import swanlab
from datetime import date


def setup_environment(seed):
    """Setup random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_base_model(model):
    """Get the base model, handling DataParallel/DDP wrapper"""
    from torch.nn.parallel import DistributedDataParallel as DDP
    if isinstance(model, (DataParallel, DDP)):
        return model.module
    return model


def save_checkpoint(model, optimizer, epoch, best_metric, args, path, vis_data=None):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": get_base_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "args": vars(args),
    }
    if vis_data is not None:
        checkpoint["visualization_images"] = vis_data.cpu()
    torch.save(checkpoint, path)


def setup_logging(args, task_name="classification"):
    """Setup SwanLab logging"""
    import os
    
    # Support both use_daga (original) and method (comparison) styles
    if hasattr(args, 'method'):
        # Paper comparison style
        method_name = args.method
        layers_str = '-'.join(map(str, args.adaptation_layers))
    else:
        # Original style
        method_name = "daga" if args.use_daga else "baseline"
        layers_str = '-'.join(map(str, args.daga_layers)) if args.use_daga else ''
    
    exp_name = (
        args.swanlab_name
        or f"{args.dataset}_{method_name}_L{layers_str}_{date.today()}"
    )
    
    # Check both environment variable and args attribute
    swanlab_mode = os.environ.get('SWANLAB_MODE', '').lower()
    enable_swanlab = getattr(args, 'enable_swanlab', True) and swanlab_mode != 'disabled'
    
    # Store enable_swanlab in args for later use
    args.enable_swanlab = enable_swanlab
    
    if enable_swanlab:
        # Define project names for different tasks
        # Paper comparison experiments use the same projects as main experiments
        project_mapping = {
            "classification": "DINOv3-ImageNet-Classification",
            "dinotxt": "DINOv3-COCO-TextImageAlignment",
            "detection": "DINOv3-COCO-Detection", 
            "segmentation": "DINOv3-ADE20K-Segmentation",
            "depth": "DINOv3-NYUv2-Depth",
            "robustness": "DINOv3-ImageNet-C-Robustness",
            "linear": "DINOv3-Linear-Probing",
            "logreg": "DINOv3-Logistic-Regression",
            "knn": "DINOv3-KNN-Evaluation",
            "paper_comparison": "DINOv3-ImageNet-Classification",  # Use same as classification
        }
        
        project_name = project_mapping.get(task_name, f"DINOv3-{task_name}")
        
        # Get mode from environment or args
        mode = swanlab_mode if swanlab_mode else getattr(args, 'swanlab_mode', 'cloud')
        
        swanlab.init(
            workspace="NUDT_SSL__CVPR",
            project=project_name,
            experiment_name=exp_name,
            config=vars(args),
            mode=mode,
        )
    return exp_name


def create_dataloaders(train_dataset, test_dataset, batch_size, num_workers=8):
    """Create train and test dataloaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to avoid DataParallel issues
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def finalize_experiment(best_metric, final_metric, total_time_minutes, output_dir, metric_name="Acc", enable_swanlab=True):
    """Print final results and close SwanLab"""
    print(f'\n{"="*70}\n🎉 Training Completed!\n{"="*70}')
    print(f"Total Time:       {total_time_minutes:.1f} minutes")
    print(f"Best {metric_name}:    {best_metric:.2f}%")
    print(f"Final {metric_name}:   {final_metric:.2f}%")
    print(f"Results saved to: {output_dir}\n{'='*70}\n")
    if enable_swanlab:
        swanlab.finish()

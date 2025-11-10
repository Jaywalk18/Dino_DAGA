import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
from pathlib import Path
import warnings
import os

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging, finalize_experiment
from core.ddp_utils import setup_ddp, cleanup_ddp, create_ddp_dataloaders
from data.segmentation_datasets import get_segmentation_dataset
from tasks.segmentation import (
    SegmentationModel, 
    setup_training_components, 
    run_training_loop,
    prepare_visualization_data
)

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DINOv3 Segmentation with DAGA")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="dinov3_vits16",
        help="DINOv3 model architecture",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        help="Path to pretrained checkpoint",
    )
    
    parser.add_argument(
        "--dataset",
        choices=["ade20k", "coco"],
        default="ade20k",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset root",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (for quick testing)",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help="Fraction (0, 1] of the dataset to use for quick testing",
    )
    
    parser.add_argument(
        "--use_daga",
        action="store_true",
        help="Use DAGA",
    )
    parser.add_argument(
        "--daga_layers",
        type=int,
        nargs="+",
        default=[11],
        help="Layers to apply DAGA",
    )
    parser.add_argument(
        "--out_indices",
        type=int,
        nargs="+",
        default=[2, 5, 8, 11],
        help="Feature extraction layers",
    )
    
    parser.add_argument("--input_size", type=int, default=518, help="Input image size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--enable_swanlab", action="store_true", default=True, help="Enable SwanLab logging")
    parser.add_argument("--swanlab_name", type=str, default=None, help="SwanLab experiment name")
    parser.add_argument("--log_freq", type=int, default=5, help="Visualization log frequency")
    parser.add_argument(
        "--enable_visualization",
        action="store_true",
        help="Enable segmentation visualization",
    )
    parser.add_argument(
        "--num_vis_samples",
        type=int,
        default=4,
        help="Number of samples to visualize",
    )
    
    return parser.parse_args()


def main():
    # Setup DDP
    local_rank, rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    
    args = parse_arguments()
    setup_environment(args.seed + rank)  # Different seed per process
    
    if is_main_process:
        experiment_name = setup_logging(args, task_name="segmentation")
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        experiment_name = None
        output_dir = None
    
    # Synchronize output_dir across all processes
    if is_main_process:
        output_dir_str = str(output_dir)
    else:
        output_dir_str = None
    
    # Broadcast output_dir to all processes
    output_dir_list = [output_dir_str] if is_main_process else [None]
    dist.broadcast_object_list(output_dir_list, src=0)
    if not is_main_process:
        output_dir = Path(output_dir_list[0])
    
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"DDP Training with {world_size} GPUs")
        print(f"Loading {args.dataset.upper()} dataset...")
    
    train_dataset, val_dataset, num_classes = get_segmentation_dataset(args)
    
    limit_train = None
    limit_val = None
    if args.sample_ratio is not None:
        if not (0 < args.sample_ratio <= 1.0):
            raise ValueError("sample_ratio must be in the range (0, 1].")
        limit_train = max(1, int(len(train_dataset) * args.sample_ratio))
        limit_val = max(1, int(len(val_dataset) * args.sample_ratio))
    if args.max_samples is not None:
        limit_train = min(args.max_samples, len(train_dataset))
        limit_val = min(max(1, args.max_samples // 2), len(val_dataset))
    
    if limit_train is not None or limit_val is not None:
        import random
        random.seed(args.seed)
        if limit_train is None:
            limit_train = len(train_dataset)
        if limit_val is None:
            limit_val = len(val_dataset)
        if limit_train < len(train_dataset):
            train_indices = random.sample(range(len(train_dataset)), limit_train)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        if limit_val < len(val_dataset):
            val_indices = random.sample(range(len(val_dataset)), limit_val)
            val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        if is_main_process:
            print(f"✓ Dataset loaded: {len(train_dataset)} train (limited), {len(val_dataset)} val (limited) samples")
    else:
        if is_main_process:
            print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    if is_main_process:
        print(f"  Number of classes: {num_classes}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Effective batch size: {args.batch_size * world_size}")
    
    train_loader, val_loader = create_ddp_dataloaders(
        train_dataset, val_dataset, args.batch_size, world_size, rank,
        num_workers=args.num_workers
    )
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Loading DINOv3 model '{args.model_name}'...")
    
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    model = SegmentationModel(
        vit_model,
        num_classes=num_classes,
        use_daga=args.use_daga,
        daga_layers=args.daga_layers,
        out_indices=args.out_indices,
    )
    
    model.to(device)
    
    # Wrap model with DDP
    # If DAGA layers are configured properly (all affect at least one layer in out_indices),
    # we don't need find_unused_parameters=True
    model = DDP(
        model, 
        device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=False,  # All DAGA layers should affect extracted features
        broadcast_buffers=False,  # Reduce communication overhead
        gradient_as_bucket_view=True  # More efficient gradient handling
    )
    
    if is_main_process:
        mode_status = "DAGA mode" if args.use_daga else "Baseline (frozen backbone)"
        print(f"\n✓ Model wrapped with DDP on {world_size} GPUs ({mode_status})")
    
    criterion, optimizer, scheduler = setup_training_components(model, args)
    
    # Only prepare visualization on main process
    if is_main_process:
        fixed_vis_images, fixed_vis_masks = prepare_visualization_data(val_dataset, args, device)
    else:
        fixed_vis_images, fixed_vis_masks = None, None
    
    if is_main_process:
        print(f"\n{'='*70}")
        print("Starting training...")
        print(f"{'='*70}\n")
    
    try:
        best_miou, final_miou, total_time = run_training_loop(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            args,
            output_dir,
            fixed_vis_images,
            fixed_vis_masks,
            num_classes,
            rank=rank,
            world_size=world_size,
        )
        
        if is_main_process:
            finalize_experiment(best_miou, final_miou, total_time, output_dir, metric_name="mIoU", enable_swanlab=args.enable_swanlab)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()

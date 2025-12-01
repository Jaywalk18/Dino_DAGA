"""
DINOv3 Semantic Segmentation with DAGA
Supports VOC2012 and Cityscapes datasets
"""
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
from core.datasets.segmentation_datasets import get_segmentation_dataset
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
    parser = argparse.ArgumentParser(description="DINOv3 Semantic Segmentation with DAGA")
    
    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="dinov3_vitb16",
        help="DINOv3 model architecture",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        help="Path to pretrained checkpoint",
    )
    
    # Dataset
    parser.add_argument(
        "--dataset",
        choices=["voc2012", "cityscapes"],
        default="voc2012",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset root",
    )
    parser.add_argument(
        "--sample_ratio", 
        type=float, 
        default=1.0, 
        help="Ratio of training data to use"
    )
    
    # DAGA
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
        help="Output indices for multi-scale features",
    )
    
    # Training
    parser.add_argument("--input_size", type=int, default=512, help="Input image size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    
    # Output
    parser.add_argument("--output_dir", default="./outputs/segmentation", help="Output directory")
    parser.add_argument("--enable_swanlab", action="store_true", default=True, help="Enable SwanLab logging")
    parser.add_argument("--swanlab_name", type=str, default=None, help="SwanLab experiment name")
    parser.add_argument("--log_freq", type=int, default=5, help="Visualization log frequency")
    parser.add_argument(
        "--num_vis_samples",
        type=int,
        default=4,
        help="Number of samples for visualization",
    )
    parser.add_argument(
        "--enable_visualization",
        action="store_true",
        help="Enable segmentation visualization",
    )
    
    return parser.parse_args()


def main():
    # Setup DDP
    local_rank, rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    args = parse_arguments()
    setup_environment(args.seed + rank)
    
    if is_main_process:
        experiment_name = setup_logging(args, task_name="segmentation")
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Broadcast output_dir
    output_dir_list = [str(output_dir)] if is_main_process else [None]
    dist.barrier()
    dist.broadcast_object_list(output_dir_list, src=0)
    if not is_main_process:
        output_dir = Path(output_dir_list[0])
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"DDP Training with {world_size} GPUs")
        print(f"Rank {rank} using device: {device}")
        print(f"Loading {args.dataset.upper()} dataset...")
    
    # Load datasets
    train_dataset, num_classes = get_segmentation_dataset(
        args.dataset, args.data_path, 'train', args.input_size
    )
    val_dataset, _ = get_segmentation_dataset(
        args.dataset, args.data_path, 'val', args.input_size
    )
    
    # Apply sample ratio if needed
    if args.sample_ratio < 1.0:
        num_samples = int(len(train_dataset) * args.sample_ratio)
        indices = torch.randperm(len(train_dataset))[:num_samples].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    if is_main_process:
        print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        print(f"  Num classes: {num_classes}")
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
    model = DDP(
        model, 
        device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True
    )
    
    if is_main_process:
        mode_status = "DAGA mode" if args.use_daga else "Baseline (frozen backbone)"
        print(f"\n✓ Model wrapped with DDP on {world_size} GPUs ({mode_status})")
    
    criterion, optimizer, scheduler = setup_training_components(model, args)
    
    # Prepare visualization data
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
            finalize_experiment(best_miou, final_miou, total_time, output_dir, 
                              metric_name="mIoU", enable_swanlab=args.enable_swanlab)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()

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
from data.classification_datasets import get_classification_dataset
from tasks.classification import (
    ClassificationModel, 
    setup_training_components, 
    run_training_loop,
    prepare_visualization_data
)

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DINOv3 Classification with DAGA")
    
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
        choices=["cifar10", "cifar100", "imagenet100", "imagenet", "food101", 
                 "flowers102", "pets", "cars", "sun397", "dtd"],
        default="cifar100",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to dataset root",
    )
    parser.add_argument(
        "--subset_ratio", 
        type=float, 
        default=1.0, 
        help="Ratio of training data to use"
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
    
    parser.add_argument("--input_size", type=int, default=518, help="Input image size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--enable_swanlab", action="store_true", default=True, help="Enable SwanLab logging")
    parser.add_argument("--swanlab_name", type=str, default=None, help="SwanLab experiment name")
    parser.add_argument("--log_freq", type=int, default=5, help="Visualization log frequency")
    parser.add_argument(
        "--vis_indices",
        type=int,
        nargs="+",
        default=[1000, 2000, 3000, 4000],
        help="Test set image indices for visualization",
    )
    parser.add_argument(
        "--enable_visualization",
        action="store_true",
        help="Enable attention visualization",
    )
    parser.add_argument(
        "--vis_attn_layer",
        type=int,
        default=11,
        help="Layer to visualize attention",
    )
    
    return parser.parse_args()


def main():
    # Setup DDP - must be first!
    local_rank, rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    
    # Ensure CUDA device is set correctly before ANY CUDA operations
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Parse arguments AFTER DDP setup
    args = parse_arguments()
    setup_environment(args.seed + rank)
    
    if is_main_process:
        experiment_name = setup_logging(args, task_name="classification")
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Broadcast output_dir to all processes
    output_dir_list = [str(output_dir)] if is_main_process else [None]
    dist.barrier()  # Add barrier before broadcast
    dist.broadcast_object_list(output_dir_list, src=0)
    if not is_main_process:
        output_dir = Path(output_dir_list[0])
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"DDP Training with {world_size} GPUs")
        print(f"Rank {rank} using device: {device}")
        print(f"Loading {args.dataset.upper()} dataset...")
    
    train_dataset, test_dataset, num_classes = get_classification_dataset(args)
    
    if is_main_process:
        print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Effective batch size: {args.batch_size * world_size}")
    
    train_loader, test_loader = create_ddp_dataloaders(
        train_dataset, test_dataset, args.batch_size, world_size, rank,
        num_workers=args.num_workers
    )
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Loading DINOv3 model '{args.model_name}'...")
    
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    model = ClassificationModel(
        vit_model,
        num_classes=num_classes,
        use_daga=args.use_daga,
        daga_layers=args.daga_layers,
        enable_visualization=args.enable_visualization,
        vis_attn_layer=args.vis_attn_layer,
    )
    
    model.to(device)
    
    # Wrap model with DDP
    # DAGA mode: if DAGA is applied to all layers in each forward pass, no unused parameters
    # Baseline mode: backbone frozen but all parameters are used in forward pass
    # In both cases, find_unused_parameters can be False for better performance
    model = DDP(
        model, 
        device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=False,  # All parameters are used in forward pass
        broadcast_buffers=False,  # Reduce communication overhead
        gradient_as_bucket_view=True  # More efficient gradient handling
    )
    
    if is_main_process:
        mode_status = "DAGA mode" if args.use_daga else "Baseline (frozen backbone)"
        print(f"\n✓ Model wrapped with DDP on {world_size} GPUs ({mode_status})")
    
    # Pass world_size for proper learning rate scaling in DDP
    criterion, optimizer, scheduler = setup_training_components(model, args)
    
    # Only prepare visualization on main process
    if is_main_process:
        fixed_vis_images = prepare_visualization_data(test_dataset, args, device)
    else:
        fixed_vis_images = None
    
    if is_main_process:
        print(f"\n{'='*70}")
        print("Starting training...")
        print(f"{'='*70}\n")
    
    try:
        best_acc, final_acc, total_time = run_training_loop(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            args,
            output_dir,
            fixed_vis_images,
            test_dataset=test_dataset,
            rank=rank,
            world_size=world_size,
        )
        
        if is_main_process:
            finalize_experiment(best_acc, final_acc, total_time, output_dir, metric_name="Acc", enable_swanlab=args.enable_swanlab)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()

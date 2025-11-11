"""
Depth Estimation Script for DINOv3 with DAGA Support
Uses official DINOv3 depth estimation architecture (DPT head)
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
from pathlib import Path
import warnings
import os
import sys
import numpy as np
from tqdm import tqdm

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging, finalize_experiment
from core.ddp_utils import setup_ddp, cleanup_ddp

# Add dinov3 to path
dinov3_path = '/home/user/zhoutianjian/Dino_DAGA/dinov3'
if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)

from dinov3.eval.dense.depth.models import build_depther
from dinov3.eval.dense.depth.models.encoder import BackboneLayersSet

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


class DepthModel(nn.Module):
    """Depth estimation model with DPT head and optional DAGA"""
    def __init__(self, vit_model, out_indices, use_daga=False, daga_layers=None,
                 min_depth=0.001, max_depth=10.0):
        super().__init__()
        self.vit_model = vit_model
        self.use_daga = use_daga
        self.daga_layers = daga_layers if daga_layers else []
        self.out_indices = out_indices
        
        # Build DPT depth head using official implementation
        self.depther = build_depther(
            backbone=vit_model,
            backbone_out_layers=out_indices,
            n_output_channels=256,
            use_backbone_norm=True,
            use_batchnorm=True,
            use_cls_token=False,
            head_type="dpt",
            min_depth=min_depth,
            max_depth=max_depth,
            channels=256,
            post_process_channels=[256, 512, 1024, 1024],
        )
    
    def forward(self, x):
        """Forward pass for depth estimation"""
        return self.depther(x)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DINOv3 Depth Estimation with DAGA")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16", help="DINOv3 model architecture")
    parser.add_argument("--pretrained_path", type=str, default="dinov3_vitb16_pretrain_lvd1689m.pth", help="Path to pretrained checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset", choices=["nyu_depth_v2", "kitti"], default="nyu_depth_v2", help="Dataset to use")
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--sample_ratio", type=float, default=None, help="Ratio of training data to use")
    
    # DAGA arguments
    parser.add_argument("--use_daga", action="store_true", help="Use DAGA")
    parser.add_argument("--daga_layers", type=int, nargs="+", default=[11], help="Layers to apply DAGA")
    
    # Depth-specific arguments
    parser.add_argument("--out_indices", type=int, nargs="+", default=[2, 5, 8, 11], help="Output layer indices for multi-scale features")
    parser.add_argument("--min_depth", type=float, default=0.001, help="Minimum depth value")
    parser.add_argument("--max_depth", type=float, default=10.0, help="Maximum depth value")
    
    # Training arguments
    parser.add_argument("--input_size", type=int, default=518, help="Input image size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Output arguments
    parser.add_argument("--output_dir", default="./outputs/depth", help="Output directory")
    parser.add_argument("--enable_visualization", action="store_true", help="Enable depth map visualization")
    parser.add_argument("--num_vis_samples", type=int, default=4, help="Number of samples to visualize")
    parser.add_argument("--log_freq", type=int, default=5, help="Logging frequency")
    
    return parser.parse_args()


def create_dummy_depth_dataset(num_samples=100, input_size=518):
    """
    Create a dummy depth dataset for demonstration
    In practice, replace this with actual dataset loading
    """
    class DummyDepthDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, input_size):
            self.num_samples = num_samples
            self.input_size = input_size
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Generate dummy RGB image
            image = torch.randn(3, self.input_size, self.input_size)
            # Generate dummy depth map (H, W)
            depth = torch.rand(self.input_size, self.input_size) * 10.0
            return image, depth
    
    return DummyDepthDataset(num_samples, input_size)


def compute_depth_metrics(pred_depth, gt_depth, min_depth=0.001, max_depth=10.0):
    """
    Compute standard depth estimation metrics
    """
    # Mask valid depth values
    mask = (gt_depth > min_depth) & (gt_depth < max_depth)
    
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]
    
    if len(pred_depth) == 0:
        return {}
    
    # Absolute relative error
    abs_rel = torch.mean(torch.abs(pred_depth - gt_depth) / gt_depth)
    
    # Squared relative error
    sq_rel = torch.mean(((pred_depth - gt_depth) ** 2) / gt_depth)
    
    # RMSE
    rmse = torch.sqrt(torch.mean((pred_depth - gt_depth) ** 2))
    
    # RMSE log
    rmse_log = torch.sqrt(torch.mean((torch.log(pred_depth) - torch.log(gt_depth)) ** 2))
    
    # Threshold accuracy
    thresh = torch.maximum(pred_depth / gt_depth, gt_depth / pred_depth)
    delta1 = (thresh < 1.25).float().mean()
    delta2 = (thresh < 1.25 ** 2).float().mean()
    delta3 = (thresh < 1.25 ** 3).float().mean()
    
    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'delta1': delta1.item(),
        'delta2': delta2.item(),
        'delta3': delta3.item(),
    }


def train_one_epoch(model, train_loader, criterion, optimizer, device, args, epoch, rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    is_main_process = (rank == 0)
    
    if is_main_process:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, (images, depths) in enumerate(pbar):
        images = images.to(device)
        depths = depths.to(device).unsqueeze(1)  # Add channel dimension
        
        # Forward pass
        pred_depths = model(images)
        
        # Compute loss
        loss = criterion(pred_depths, depths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if is_main_process:
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, test_loader, device, args, rank):
    """Evaluate the model"""
    model.eval()
    metrics_sum = {}
    num_batches = 0
    
    is_main_process = (rank == 0)
    
    with torch.no_grad():
        if is_main_process:
            pbar = tqdm(test_loader, desc="Evaluating")
        else:
            pbar = test_loader
        
        for images, depths in pbar:
            images = images.to(device)
            depths = depths.to(device)
            
            # Forward pass
            pred_depths = model(images).squeeze(1)
            
            # Compute metrics
            batch_metrics = compute_depth_metrics(
                pred_depths, depths, args.min_depth, args.max_depth
            )
            
            for key, val in batch_metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + val
            
            num_batches += 1
    
    # Average metrics
    metrics_avg = {k: v / num_batches for k, v in metrics_sum.items()}
    return metrics_avg


def main():
    # Setup DDP
    local_rank, rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Parse arguments
    args = parse_arguments()
    setup_environment(args.seed + rank)
    
    if is_main_process:
        experiment_name = setup_logging(args, task_name="depth")
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Depth Estimation with {world_size} GPUs")
        print(f"DAGA: {'Enabled' if args.use_daga else 'Disabled'}")
        if args.use_daga:
            print(f"DAGA Layers: {args.daga_layers}")
        print(f"{'='*70}\n")
    
    # Broadcast output_dir to all processes
    output_dir_list = [str(output_dir)] if is_main_process else [None]
    dist.barrier()
    dist.broadcast_object_list(output_dir_list, src=0)
    if not is_main_process:
        output_dir = Path(output_dir_list[0])
    
    # Load model
    if is_main_process:
        print(f"Loading DINOv3 model '{args.model_name}'...")
    
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    # Create depth model
    model = DepthModel(
        vit_model,
        out_indices=args.out_indices,
        use_daga=args.use_daga,
        daga_layers=args.daga_layers,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    model.to(device)
    
    # Wrap with DDP
    # Determine if we need find_unused_parameters based on DAGA configuration
    need_find_unused = False
    if args.use_daga and args.daga_layers:
        max_out_idx = max(args.out_indices)
        # If all DAGA layers are after the last output index, they won't affect output
        if all(daga_idx > max_out_idx for daga_idx in args.daga_layers):
            need_find_unused = True
    
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=need_find_unused,
    )
    
    if is_main_process:
        print(f"✓ Model loaded and wrapped with DDP\n")
    
    # Create datasets (using dummy data for now)
    if is_main_process:
        print("Creating datasets...")
    
    train_dataset = create_dummy_depth_dataset(100, args.input_size)
    test_dataset = create_dummy_depth_dataset(20, args.input_size)
    
    # Create dataloaders with DDP samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Setup training
    criterion = nn.L1Loss()  # MAE loss for depth
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr * world_size,  # Scale LR with world size
        weight_decay=args.weight_decay
    )
    
    if is_main_process:
        print(f"\n{'='*70}")
        print("Starting training...")
        print(f"{'='*70}\n")
    
    best_metric = float('inf')
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        # Train
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args, epoch, rank
        )
        
        # Evaluate
        if (epoch + 1) % args.log_freq == 0:
            metrics = evaluate(model, test_loader, device, args, rank)
            
            if is_main_process:
                print(f"\nEpoch {epoch+1}/{args.epochs}:")
                print(f"  Train Loss: {avg_loss:.4f}")
                print(f"  Metrics:")
                for key, val in metrics.items():
                    print(f"    {key}: {val:.4f}")
                
                # Save best model
                if metrics['abs_rel'] < best_metric:
                    best_metric = metrics['abs_rel']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics,
                    }, output_dir / "best_model.pth")
                    print(f"  ✓ Saved best model (abs_rel: {best_metric:.4f})")
        
        dist.barrier()
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Best abs_rel: {best_metric:.4f}")
        print(f"{'='*70}\n")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()


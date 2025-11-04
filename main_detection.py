import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import argparse
from pathlib import Path
import warnings

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging, finalize_experiment
from data.detection_datasets import get_detection_dataset, detection_collate_fn
from torch.utils.data import DataLoader
from tasks.detection import (
    DetectionModel, 
    setup_training_components, 
    run_training_loop,
    prepare_visualization_data
)

warnings.filterwarnings("ignore")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DINOv3 Detection with DAGA")
    
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
        choices=["coco"],
        default="coco",
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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--enable_swanlab", action="store_true", default=True, help="Enable SwanLab logging")
    parser.add_argument("--swanlab_name", type=str, default=None, help="SwanLab experiment name")
    parser.add_argument("--log_freq", type=int, default=5, help="Visualization log frequency")
    parser.add_argument(
        "--enable_visualization",
        action="store_true",
        help="Enable detection visualization",
    )
    parser.add_argument(
        "--num_vis_samples",
        type=int,
        default=4,
        help="Number of samples to visualize",
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_environment(args.seed)
    
    experiment_name = setup_logging(args, task_name="detection")
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"Loading {args.dataset.upper()} dataset...")
    train_dataset, val_dataset, num_classes = get_detection_dataset(args)
    
    # Apply max_samples limit if specified
    if args.max_samples is not None and args.max_samples < len(train_dataset):
        import random
        random.seed(args.seed)
        indices = random.sample(range(len(train_dataset)), args.max_samples)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"✓ Dataset loaded: {len(train_dataset)} train (limited), {len(val_dataset)} val samples")
    else:
        print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    print(f"  Number of classes: {num_classes}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=detection_collate_fn,
    )
    
    print(f"\n{'='*70}")
    print(f"Loading DINOv3 model '{args.model_name}'...")
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    model = DetectionModel(
        vit_model,
        num_classes=num_classes,
        use_daga=args.use_daga,
        daga_layers=args.daga_layers,
    )
    
    model.to(device)
    print(f"\n✓ Model moved to device: {device}")
    
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    optimizer, scheduler = setup_training_components(model, args)
    
    fixed_vis_images, fixed_vis_boxes = prepare_visualization_data(val_dataset, args, device)
    
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    best_loss, final_metrics, total_time = run_training_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        args,
        output_dir,
        fixed_vis_images,
        fixed_vis_boxes,
    )
    
    print(f'\n{"="*70}\n🎉 Training Completed!\n{"="*70}')
    print(f"Total Time:       {total_time:.1f} minutes")
    print(f"Best Val Loss:    {best_loss:.4f}")
    print(f"Final Val Loss:   {final_metrics['loss']:.4f}")
    print(f"Final mAP:        {final_metrics['mAP']:.2f}%")
    print(f"Final Precision:  {final_metrics['precision']:.2f}%")
    print(f"Final Recall:     {final_metrics['recall']:.2f}%")
    print(f"Results saved to: {output_dir}\n{'='*70}\n")
    
    if args.enable_swanlab:
        import swanlab
        swanlab.finish()


if __name__ == "__main__":
    main()

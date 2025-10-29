import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import argparse
from pathlib import Path
import warnings

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging, create_dataloaders, finalize_experiment
from data.classification_datasets import get_classification_dataset
from tasks.classification import (
    ClassificationModel, 
    setup_training_components, 
    run_training_loop,
    prepare_visualization_data
)

warnings.filterwarnings("ignore")
import os
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
        choices=["cifar10", "cifar100", "imagenet100", "imagenet"],
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
    
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--swanlab_name", type=str, default=None, help="SwanLab experiment name")
    parser.add_argument("--log_freq", type=int, default=5, help="Visualization log frequency")
    parser.add_argument(
        "--vis_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Image indices for visualization",
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
    args = parse_arguments()
    setup_environment(args.seed)
    
    experiment_name = setup_logging(args, task_name="classification")
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"Loading {args.dataset.upper()} dataset...")
    train_dataset, test_dataset, num_classes = get_classification_dataset(args)
    print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    train_loader, test_loader = create_dataloaders(
        train_dataset, test_dataset, args.batch_size
    )
    
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
    print(f"\n✓ Model moved to device: {device}")
    
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    criterion, optimizer, scheduler = setup_training_components(model, args)
    
    fixed_vis_images = prepare_visualization_data(test_dataset, args, device)
    
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
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
    )
    
    finalize_experiment(best_acc, final_acc, total_time, output_dir, metric_name="Acc")


if __name__ == "__main__":
    main()

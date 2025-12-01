"""
DINOv3 Instance Retrieval with DAGA
Evaluates on Revisited Oxford (ROxford) and Revisited Paris (RParis) datasets
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
from core.ddp_utils import setup_ddp, cleanup_ddp
from core.datasets.retrieval_datasets import get_retrieval_dataset
from tasks.retrieval import (
    RetrievalModel,
    run_retrieval_evaluation,
    setup_training_components,
    train_with_contrastive_loss,
    visualize_retrieval_results,
)

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DINOv3 Instance Retrieval with DAGA")
    
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
        choices=["roxford5k", "rparis6k"],
        default="roxford5k",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset root",
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
    
    # Feature extraction
    parser.add_argument(
        "--pooling",
        choices=["cls", "avg", "gem"],
        default="gem",
        help="Feature pooling method",
    )
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    
    # Training (for DAGA)
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (for DAGA)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    
    # Output
    parser.add_argument("--output_dir", default="./outputs/retrieval", help="Output directory")
    parser.add_argument("--enable_swanlab", action="store_true", default=True, help="Enable SwanLab logging")
    parser.add_argument("--swanlab_name", type=str, default=None, help="SwanLab experiment name")
    parser.add_argument(
        "--enable_visualization",
        action="store_true",
        help="Enable retrieval visualization",
    )
    parser.add_argument(
        "--num_vis_queries",
        type=int,
        default=5,
        help="Number of queries to visualize",
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
        experiment_name = setup_logging(args, task_name="retrieval")
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
        print(f"DINOv3 Instance Retrieval Evaluation")
        print(f"Rank {rank} using device: {device}")
        print(f"Loading {args.dataset.upper()} dataset...")
    
    # Load datasets
    db_dataset, query_dataset = get_retrieval_dataset(
        args.dataset, args.data_path, args.input_size
    )
    
    if is_main_process:
        print(f"✓ Dataset loaded:")
        print(f"  - Database: {len(db_dataset)} images")
        print(f"  - Queries: {len(query_dataset)} images")
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Loading DINOv3 model '{args.model_name}'...")
    
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    model = RetrievalModel(
        vit_model,
        use_daga=args.use_daga,
        daga_layers=args.daga_layers,
        pooling=args.pooling,
    )
    
    model.to(device)
    
    # For retrieval, we typically don't need DDP since it's evaluation
    # But wrap for consistency if using DAGA training
    if args.use_daga and args.epochs > 0:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank, 
            find_unused_parameters=False,
        )
        
        if is_main_process:
            print(f"\n✓ Model wrapped with DDP for DAGA training")
        
        # Setup training
        optimizer, scheduler = setup_training_components(model, args)
        
        if optimizer is not None:
            if is_main_process:
                print(f"\n{'='*70}")
                print("Training DAGA with contrastive loss...")
            
            train_with_contrastive_loss(
                model, db_dataset, query_dataset,
                optimizer, scheduler, device, args, output_dir, rank
            )
    
    if is_main_process:
        print(f"\n{'='*70}")
        print("Running retrieval evaluation...")
    
    # Run evaluation
    results = run_retrieval_evaluation(
        model, db_dataset, query_dataset,
        device, args, output_dir, rank
    )
    
    # Visualize results
    if is_main_process and args.enable_visualization:
        print("\n📊 Generating visualizations...")
        visualize_retrieval_results(
            model, db_dataset, query_dataset,
            args, output_dir,
            num_queries=args.num_vis_queries
        )
    
    # Save results
    if is_main_process:
        results_file = output_dir / "results.txt"
        with open(results_file, 'w') as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"DAGA: {args.use_daga}\n")
            f.write(f"Pooling: {args.pooling}\n\n")
            
            for protocol in ['easy', 'medium', 'hard']:
                f.write(f"{protocol.upper()} Protocol:\n")
                f.write(f"  mAP: {results[protocol]['mAP']:.2f}%\n")
                for k, v in results[protocol]['recalls'].items():
                    f.write(f"  R@{k}: {v:.2f}%\n")
                f.write("\n")
        
        print(f"\n✅ Results saved to {results_file}")
        
        # Final summary
        print(f"\n{'='*70}")
        print("📊 Final Results Summary")
        print(f"{'='*70}")
        print(f"Dataset: {args.dataset}")
        print(f"DAGA: {'Yes' if args.use_daga else 'No'}")
        print(f"\nMedium Protocol (Standard):")
        print(f"  mAP: {results['medium']['mAP']:.2f}%")
        print(f"  R@1: {results['medium']['recalls'][1]:.2f}%")
        print(f"  R@10: {results['medium']['recalls'][10]:.2f}%")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()


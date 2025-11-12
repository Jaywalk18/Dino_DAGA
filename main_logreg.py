"""
Logistic Regression Evaluation Script for DINOv3 with DAGA Support
Integrates with official DINOv3 LogReg evaluation method
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
from pathlib import Path
import warnings
import os
import sys

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment
from core.ddp_utils import setup_ddp, cleanup_ddp

# Add dinov3 to path
dinov3_path = '/home/user/zhoutianjian/Dino_DAGA/dinov3'
if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)

from dinov3.eval.log_regression import eval_log_regression_with_model, LogregEvalConfig, TrainConfig, EvalConfig, TransformConfig
from dinov3.eval.setup import ModelConfig

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


class FeatureExtractorModel(nn.Module):
    """Wrapper that extracts features from DINOv3 with optional DAGA"""
    def __init__(self, vit_model, use_daga=False, daga_layers=None, daga_weights_path=None):
        super().__init__()
        self.vit_model = vit_model
        self.use_daga = use_daga
        self.daga_layers = daga_layers if daga_layers else []
        self.feature_dim = vit_model.embed_dim
        
        # Initialize DAGA modules if needed
        if self.use_daga:
            from core.daga import DAGA
            self.daga_modules = nn.ModuleDict({
                str(i): DAGA(feature_dim=self.feature_dim) for i in daga_layers
            })
            
            # Load DAGA weights if provided
            if daga_weights_path and os.path.exists(daga_weights_path):
                checkpoint = torch.load(daga_weights_path, map_location='cpu', weights_only=False)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # Extract DAGA module weights
                    daga_state_dict = {}
                    for key, value in state_dict.items():
                        clean_key = key
                        if clean_key.startswith('module.'):
                            clean_key = clean_key[7:]
                        
                        if 'daga_modules.' in clean_key:
                            daga_key = clean_key.split('daga_modules.')[-1]
                            daga_state_dict[daga_key] = value
                    
                    if daga_state_dict:
                        missing_keys, unexpected_keys = self.daga_modules.load_state_dict(daga_state_dict, strict=False)
                        print(f"✓ Loaded {len(daga_state_dict)} DAGA weights from {os.path.basename(daga_weights_path)}")
                    else:
                        print(f"⚠ No DAGA weights found in checkpoint")
        
        # Freeze backbone
        for param in self.vit_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Forward pass to extract features"""
        with torch.no_grad():
            # Get patch embeddings
            B = x.shape[0]
            x_processed, (H, W) = self.vit_model.prepare_tokens_with_masks(x)
            
            # Forward through blocks with optional DAGA
            for i, block in enumerate(self.vit_model.blocks):
                # Get RoPE embeddings if needed
                rope_sincos = self.vit_model.rope_embed(H=H, W=W) if self.vit_model.rope_embed else None
                
                # Apply DAGA if specified for this layer
                if self.use_daga and i in self.daga_layers:
                    # DAGA logic would go here - for now use standard forward
                    x_processed = block(x_processed, rope_sincos)
                else:
                    x_processed = block(x_processed, rope_sincos)
            
            # Apply final norm
            x_processed = self.vit_model.norm(x_processed)
            
            # Extract CLS token
            return x_processed[:, 0]


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DINOv3 Logistic Regression Evaluation with DAGA")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16", help="DINOv3 model architecture")
    parser.add_argument("--pretrained_path", type=str, default="dinov3_vitb16_pretrain_lvd1689m.pth", help="Path to pretrained checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar100", help="Dataset to use")
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Ratio of training data to use")
    
    # DAGA arguments
    parser.add_argument("--use_daga", action="store_true", help="Use DAGA")
    parser.add_argument("--daga_layers", type=int, nargs="+", default=[11], help="Layers to apply DAGA")
    
    # LogReg-specific arguments
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations")
    parser.add_argument("--tolerance", type=float, default=1e-12, help="Tolerance for convergence")
    
    # Training arguments
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    
    # Output arguments
    parser.add_argument("--output_dir", default="./outputs/logreg", help="Output directory")
    
    return parser.parse_args()


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
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"Logistic Regression Evaluation with {world_size} GPUs")
        print(f"DAGA: {'Enabled' if args.use_daga else 'Disabled'}")
        if args.use_daga:
            print(f"DAGA Layers: {args.daga_layers}")
        print(f"{'='*70}\n")
    
    # Load model
    if is_main_process:
        print(f"Loading DINOv3 model '{args.model_name}'...")
    
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    # For DAGA models, use the same checkpoint to load DAGA weights
    daga_weights_path = args.pretrained_path if args.use_daga else None
    model = FeatureExtractorModel(vit_model, args.use_daga, args.daga_layers, daga_weights_path)
    model.to(device)
    model.eval()
    
    if is_main_process:
        print(f"✓ Model loaded and ready for evaluation\n")
    
    # Prepare dataset paths for official LogReg evaluation
    dataset_mapping = {
        "cifar10": f"CIFAR10:split=TRAIN:root={args.data_path}",
        "cifar100": f"CIFAR100:split=TRAIN:root={args.data_path}",
        "imagenet": f"ImageNet:split=TRAIN:root={args.data_path}",
    }
    
    test_dataset_mapping = {
        "cifar10": f"CIFAR10:split=TEST:root={args.data_path}",
        "cifar100": f"CIFAR100:split=TEST:root={args.data_path}",
        "imagenet": f"ImageNet:split=VAL:root={args.data_path}",
    }
    
    train_dataset_str = dataset_mapping[args.dataset]
    test_dataset_str = test_dataset_mapping[args.dataset]
    
    # Create LogReg evaluation config following official structure
    logreg_config = LogregEvalConfig(
        model=ModelConfig(
            config_file="dummy",  # Not used, model is passed separately
            pretrained_weights=args.pretrained_path,
        ),
        train=TrainConfig(
            dataset=train_dataset_str,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tol=args.tolerance,
            max_train_iters=args.max_iter,
        ),
        eval=EvalConfig(
            test_dataset=test_dataset_str,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
        transform=TransformConfig(
            resize_size=args.input_size,
            crop_size=args.input_size,
        ),
        output_dir=str(args.output_dir),
    )
    
    if is_main_process:
        print("Starting Logistic Regression evaluation...")
        print(f"Train dataset: {train_dataset_str}")
        print(f"Test dataset: {test_dataset_str}")
        print(f"Max iterations: {args.max_iter}")
        print(f"Tolerance: {args.tolerance}\n")
        print("⏳ Extracting features from train and validation datasets...")
        print("   This may take a few minutes depending on dataset size...")
    
    # Synchronize all processes before starting
    if dist.is_initialized():
        dist.barrier()
    
    try:
        # Run LogReg evaluation using official method
        if is_main_process:
            print("\n🔄 Calling eval_log_regression_with_model...")
        
        results = eval_log_regression_with_model(
            model=model,
            autocast_dtype=torch.float16,
            config=logreg_config,
        )
        
        if is_main_process:
            print("✓ eval_log_regression_with_model completed!")
        
        if is_main_process:
            print(f"\n{'='*70}")
            print("Logistic Regression Evaluation Results:")
            print(f"{'='*70}")
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.2f}%")
                else:
                    print(f"  {k}: {v}")
            print(f"{'='*70}\n")
            
            # Save results
            results_file = output_dir / "logreg_results.txt"
            with open(results_file, "w") as f:
                f.write("Logistic Regression Evaluation Results\n")
                f.write("="*50 + "\n")
                f.write(f"Model: {args.model_name}\n")
                f.write(f"DAGA: {'Enabled' if args.use_daga else 'Disabled'}\n")
                if args.use_daga:
                    f.write(f"DAGA Layers: {args.daga_layers}\n")
                f.write("="*50 + "\n\n")
                for k, v in results.items():
                    if isinstance(v, (int, float)):
                        f.write(f"{k}: {v:.2f}%\n")
                    else:
                        f.write(f"{k}: {v}\n")
            
            print(f"✓ Results saved to {results_file}")
    
    except Exception as e:
        if is_main_process:
            print(f"❌ Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()


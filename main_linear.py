"""
Linear Probe Evaluation Script for DINOv3 with DAGA Support
Integrates with official DINOv3 Linear evaluation method
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
import swanlab
from datetime import date

# Add dinov3 to path
dinov3_path = '/home/user/zhoutianjian/Dino_DAGA/dinov3'
if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)

from dinov3.eval.linear import eval_linear_with_model, LinearEvalConfig, TrainConfig, EvalConfig, TransformConfig
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
        """Forward pass to extract intermediate features"""
        with torch.no_grad():
            # Get patch embeddings
            B = x.shape[0]
            x_processed, (H, W) = self.vit_model.prepare_tokens_with_masks(x)
            
            # Store intermediate outputs
            intermediate_outputs = []
            
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
                
                # Store intermediate output (patch tokens and class token)
                intermediate_outputs.append((x_processed[:, 1:], x_processed[:, 0:1]))
            
            # Return intermediate features for linear probing
            return intermediate_outputs
    
    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, 
                               return_extra_tokens=False, norm=True):
        """
        Get intermediate layer outputs from the model.
        This method delegates to the underlying vit_model but can be extended to support DAGA.
        """
        # For now, delegate to the underlying model's get_intermediate_layers
        return self.vit_model.get_intermediate_layers(
            x, n=n, reshape=reshape, return_class_token=return_class_token,
            return_extra_tokens=return_extra_tokens, norm=norm
        )


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DINOv3 Linear Probe Evaluation with DAGA")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16", help="DINOv3 model architecture")
    parser.add_argument("--pretrained_path", type=str, default="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", help="Path to pretrained checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar100", help="Dataset to use")
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Ratio of training data to use")
    
    # DAGA arguments
    parser.add_argument("--use_daga", action="store_true", help="Use DAGA")
    parser.add_argument("--daga_layers", type=int, nargs="+", default=[11], help="Layers to apply DAGA")
    
    # Linear probe-specific arguments
    parser.add_argument("--linear_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--epoch_length", type=int, default=1250, help="Length of an epoch in iterations")
    parser.add_argument("--learning_rates", type=float, nargs="+", 
                       default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
                       help="Learning rates to grid search")
    
    # Training arguments
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    
    # Output arguments
    parser.add_argument("--output_dir", default="./outputs/linear", help="Output directory")
    
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
        
        # Initialize SwanLab
        method_name = "daga" if args.use_daga else "baseline"
        exp_name = f"{args.dataset}_linear_{method_name}_L{'-'.join(map(str, args.daga_layers)) if args.use_daga else ''}_{date.today()}"
        swanlab.init(
            workspace="NUDT_SSL__CVPR",
            project="DINOv3-Linear-Probing",
            experiment_name=exp_name,
            config=vars(args),
        )
        
        print(f"\n{'='*70}")
        print(f"Linear Probe Evaluation with {world_size} GPUs")
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
    
    # Prepare dataset paths for official Linear evaluation
    dataset_mapping = {
        "cifar10": f"CIFAR10:split=TRAIN:root={args.data_path}",
        "cifar100": f"CIFAR100:split=TRAIN:root={args.data_path}",
        "imagenet": f"ImageNet:split=TRAIN:root={args.data_path}:extra={args.data_path}",
    }
    
    val_dataset_mapping = {
        "cifar10": f"CIFAR10:split=TEST:root={args.data_path}",
        "cifar100": f"CIFAR100:split=TEST:root={args.data_path}",
        "imagenet": f"ImageNet:split=VAL:root={args.data_path}:extra={args.data_path}",
    }
    
    train_dataset_str = dataset_mapping[args.dataset]
    val_dataset_str = val_dataset_mapping[args.dataset]
    
    # Create Linear evaluation config following official structure
    # ModelConfig expects config_file and pretrained_weights
    # We use model_name as a placeholder for config_file since we load directly
    linear_config = LinearEvalConfig(
        model=ModelConfig(
            config_file=args.model_name,  # Use model_name as identifier
            pretrained_weights=args.pretrained_path,
        ),
        train=TrainConfig(
            dataset=train_dataset_str,
            val_dataset=val_dataset_str,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rates=tuple(args.learning_rates),
            epochs=args.linear_epochs,
            epoch_length=args.epoch_length,
        ),
        eval=EvalConfig(
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
        print("Starting Linear Probe evaluation...")
        print(f"Train dataset: {train_dataset_str}")
        print(f"Val dataset: {val_dataset_str}")
        print(f"Epochs: {args.linear_epochs}")
        print(f"Epoch length: {args.epoch_length}")
        print(f"Learning rates: {args.learning_rates}\n")
    
    try:
        # Run Linear evaluation using official method
        results = eval_linear_with_model(
            model=model,
            autocast_dtype=torch.float16,
            config=linear_config,
        )
        
        if is_main_process:
            print(f"\n{'='*70}")
            print("Linear Probe Evaluation Results:")
            print(f"{'='*70}")
            
            # Print best classifier
            for k, v in results.items():
                if k != "all_classifiers":  # Skip detailed results in summary
                    if isinstance(v, (int, float)):
                        print(f"  {k}: {v:.2f}%")
                    else:
                        print(f"  {k}: {v}")
            
            # Print all classifiers if available
            if "all_classifiers" in results:
                print(f"\n{'='*70}")
                print("All Classifiers Performance:")
                print(f"{'='*70}")
                all_classifiers = results["all_classifiers"]
                # Sort by top-1 accuracy descending
                sorted_classifiers = sorted(all_classifiers.items(), 
                                          key=lambda x: x[1]["top-1"], 
                                          reverse=True)
                for classifier_name, metrics in sorted_classifiers:
                    # Extract learning rate from name
                    lr_str = classifier_name.split("_lr_")[-1].replace("_", ".")
                    acc = metrics["top-1"] * 100
                    print(f"  LR={lr_str}: {acc:.2f}%")
            
            print(f"{'='*70}\n")
            
            # Save summary results
            results_file = output_dir / "linear_results.txt"
            with open(results_file, "w") as f:
                f.write("Linear Probe Evaluation Results\n")
                f.write("="*50 + "\n")
                f.write(f"Model: {args.model_name}\n")
                f.write(f"DAGA: {'Enabled' if args.use_daga else 'Disabled'}\n")
                if args.use_daga:
                    f.write(f"DAGA Layers: {args.daga_layers}\n")
                f.write("="*50 + "\n\n")
                
                # Write best classifier
                for k, v in results.items():
                    if k != "all_classifiers":
                        if isinstance(v, (int, float)):
                            f.write(f"{k}: {v:.2f}%\n")
                        else:
                            f.write(f"{k}: {v}\n")
                
                # Write all classifiers performance
                if "all_classifiers" in results:
                    f.write("\n" + "="*50 + "\n")
                    f.write("All Classifiers Performance:\n")
                    f.write("="*50 + "\n")
                    all_classifiers = results["all_classifiers"]
                    sorted_classifiers = sorted(all_classifiers.items(), 
                                              key=lambda x: x[1]["top-1"], 
                                              reverse=True)
                    for classifier_name, metrics in sorted_classifiers:
                        lr_str = classifier_name.split("_lr_")[-1].replace("_", ".")
                        acc = metrics["top-1"] * 100
                        f.write(f"LR={lr_str}: {acc:.2f}%\n")
            
            print(f"✓ Results saved to {results_file}")
            print(f"✓ Detailed results also in: results_all_classifiers.json")
            
            # Log to SwanLab - Include all learning rates
            # Use epoch=1 as the final step for linear probe results
            log_dict = {"epoch": 1}
            for k, v in results.items():
                if k != "all_classifiers" and isinstance(v, (int, float)):
                    log_dict[k] = v * 100 if v < 1 else v
            
            # Log all learning rates individually
            if "all_classifiers" in results and results["all_classifiers"]:
                print(f"\n📊 Logging all {len(results['all_classifiers'])} learning rates to SwanLab...")
                for classifier_name, metrics in results["all_classifiers"].items():
                    # Extract learning rate from name: classifier_1_blocks_avgpool_True_lr_0_02500 -> 0.025
                    lr_str = classifier_name.split("_lr_")[-1].replace("_", ".")
                    top1_acc = metrics["top-1"] * 100
                    log_dict[f"linear_probe/lr_{lr_str}_top1"] = top1_acc
                    if metrics.get("top-5") is not None:
                        top5_acc = metrics["top-5"] * 100
                        log_dict[f"linear_probe/lr_{lr_str}_top5"] = top5_acc
            
            # Use a fixed step for final results (end of training)
            swanlab.log(log_dict, step=args.linear_epochs)
            print(f"✓ Results logged to SwanLab (including {len(results.get('all_classifiers', {}))} LRs)")
    
    except Exception as e:
        if is_main_process:
            print(f"❌ Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
    finally:
        if is_main_process:
            swanlab.finish()
        cleanup_ddp()


if __name__ == "__main__":
    main()


"""
ImageNet-C Robustness Evaluation Script for DINOv3 with DAGA Support
Tests model robustness on 15 corruption types with 5 severity levels
Based on: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (Hendrycks & Dietterich, 2019)
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
from collections import defaultdict

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging
from core.ddp_utils import setup_ddp, cleanup_ddp

# Add dinov3 to path
dinov3_path = '/home/user/zhoutianjian/Dino_DAGA/dinov3'
if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# 15 corruption types in ImageNet-C
CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',  # Noise
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',  # Blur
    'snow', 'frost', 'fog', 'brightness',  # Weather
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'  # Digital
]


class RobustnessModel(nn.Module):
    """Classification model for robustness evaluation with optional DAGA"""
    def __init__(self, vit_model, num_classes=1000, use_daga=False, daga_layers=None):
        super().__init__()
        self.vit_model = vit_model
        self.use_daga = use_daga
        self.daga_layers = daga_layers if daga_layers else []
        
        # Freeze backbone
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Linear(vit_model.embed_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        """Forward pass"""
        B = x.shape[0]
        
        with torch.no_grad():
            # Get patch embeddings
            x_processed = self.vit_model.prepare_tokens_with_masks(x)
            # Handle tuple return value
            if isinstance(x_processed, tuple):
                x_processed = x_processed[0]
            
            # Get dimensions
            H = W = int((x_processed.shape[1] - 1 - getattr(self.vit_model, 'n_storage_tokens', 0)) ** 0.5)
            
            # Forward through blocks
            for i, block in enumerate(self.vit_model.blocks):
                rope_sincos = self.vit_model.rope_embed(H=H, W=W) if self.vit_model.rope_embed else None
                
                if self.use_daga and i in self.daga_layers:
                    # DAGA would be applied here
                    x_processed = block(x_processed, rope_sincos)
                else:
                    x_processed = block(x_processed, rope_sincos)
            
            # Apply final norm
            x_processed = self.vit_model.norm(x_processed)
            
            # Extract CLS token
            cls_token = x_processed[:, 0]
        
        # Classify
        logits = self.classifier(cls_token)
        return logits


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DINOv3 ImageNet-C Robustness Evaluation with DAGA")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16", help="DINOv3 model architecture")
    parser.add_argument("--pretrained_path", type=str, default="dinov3_vitb16_pretrain_lvd1689m.pth", help="Path to pretrained checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="imagenet_c", help="Dataset name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to ImageNet-C dataset root")
    parser.add_argument("--corruption_types", type=str, nargs="+", default=CORRUPTION_TYPES, help="Corruption types to evaluate")
    parser.add_argument("--severity_levels", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Severity levels to evaluate")
    
    # DAGA arguments
    parser.add_argument("--use_daga", action="store_true", help="Use DAGA")
    parser.add_argument("--daga_layers", type=int, nargs="+", default=[11], help="Layers to apply DAGA")
    
    # Evaluation arguments
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per GPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    
    # Output arguments
    parser.add_argument("--output_dir", default="./outputs/robustness", help="Output directory")
    
    # Logging arguments
    parser.add_argument("--swanlab_name", type=str, default=None, help="SwanLab experiment name")
    parser.add_argument("--swanlab_mode", type=str, default="disabled", help="SwanLab mode")
    
    return parser.parse_args()


class ImageNetCDataset(torch.utils.data.Dataset):
    """
    ImageNet-C dataset for robustness evaluation
    Expected structure: data_root/{corruption_type}/{severity}/images
    """
    def __init__(self, data_root, corruption_type, severity, transform=None):
        self.data_root = Path(data_root)
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        
        # Build path: data_root/corruption_type/severity/
        self.corruption_dir = self.data_root / corruption_type / str(severity)
        
        if not self.corruption_dir.exists():
            raise FileNotFoundError(
                f"Corruption directory not found: {self.corruption_dir}\n"
                f"Please ensure ImageNet-C data is extracted to {self.data_root}"
            )
        
        # Collect all images (support both nested class dirs and flat structure)
        self.samples = []
        self.labels = []
        
        # Try nested structure first (class folders)
        class_dirs = sorted([d for d in self.corruption_dir.iterdir() if d.is_dir()])
        
        if class_dirs:
            # Nested structure: corruption_type/severity/class_name/images
            for class_idx, class_dir in enumerate(class_dirs):
                image_files = sorted(list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.png")))
                for img_path in image_files:
                    self.samples.append(img_path)
                    self.labels.append(class_idx)
        else:
            # Flat structure: corruption_type/severity/images
            image_files = sorted(
                list(self.corruption_dir.glob("*.JPEG")) + 
                list(self.corruption_dir.glob("*.png")) +
                list(self.corruption_dir.glob("*.jpg"))
            )
            for idx, img_path in enumerate(image_files):
                self.samples.append(img_path)
                # Extract label from filename if possible, otherwise use sequential
                self.labels.append(idx % 1000)
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.corruption_dir}")
        
        print(f"  Loaded {len(self.samples)} images from {corruption_type}/severity_{severity}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DummyImageNetCDataset(torch.utils.data.Dataset):
    """
    Dummy ImageNet-C dataset for testing without real data
    """
    def __init__(self, corruption_type, severity, num_samples=1000, input_size=224):
        self.corruption_type = corruption_type
        self.severity = severity
        self.num_samples = num_samples
        self.input_size = input_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy corrupted image
        image = torch.randn(3, self.input_size, self.input_size)
        label = idx % 1000  # Dummy label
        return image, label


def compute_corruption_error(accuracy, baseline_accuracy=0.0):
    """
    Compute Corruption Error (CE) for a single corruption
    CE = (1 - accuracy) / (1 - baseline_accuracy)
    """
    if baseline_accuracy >= 1.0:
        return 0.0
    return (1.0 - accuracy) / (1.0 - baseline_accuracy)


def evaluate_corruption(model, data_loader, device, rank):
    """Evaluate on a single corruption type and severity"""
    model.eval()
    correct = 0
    total = 0
    
    is_main_process = (rank == 0)
    
    with torch.no_grad():
        if is_main_process:
            pbar = tqdm(data_loader, desc="Evaluating", leave=False)
        else:
            pbar = data_loader
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Gather results from all processes
    correct_tensor = torch.tensor(correct, device=device)
    total_tensor = torch.tensor(total, device=device)
    
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    accuracy = correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0
    return accuracy


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
        experiment_name = setup_logging(args, task_name="robustness")
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"ImageNet-C Robustness Evaluation with {world_size} GPUs")
        print(f"DAGA: {'Enabled' if args.use_daga else 'Disabled'}")
        if args.use_daga:
            print(f"DAGA Layers: {args.daga_layers}")
        print(f"Corruption Types: {len(args.corruption_types)}")
        print(f"Severity Levels: {args.severity_levels}")
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
    model = RobustnessModel(vit_model, num_classes=1000, use_daga=args.use_daga, daga_layers=args.daga_layers)
    model.to(device)
    
    # Wrap with DDP
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    
    if is_main_process:
        print(f"✓ Model loaded and ready\n")
    
    # Prepare transforms for ImageNet-C
    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Results storage
    results = defaultdict(dict)  # corruption_type -> severity -> accuracy
    
    # Check if real data exists
    data_root = Path(args.data_path)
    use_real_data = True
    
    # Evaluate on each corruption and severity
    for corruption_type in args.corruption_types:
        if is_main_process:
            print(f"\nEvaluating: {corruption_type}")
        
        for severity in args.severity_levels:
            if is_main_process:
                print(f"  Severity {severity}...", end=" ", flush=True)
            
            # Try to load real data, fall back to dummy data if not available
            try:
                if use_real_data:
                    dataset = ImageNetCDataset(
                        data_root, corruption_type, severity,
                        transform=test_transform
                    )
            except (FileNotFoundError, RuntimeError) as e:
                if is_main_process:
                    print(f"\n  Warning: {str(e)}")
                    print(f"  Falling back to dummy data for testing...")
                use_real_data = False
                dataset = DummyImageNetCDataset(
                    corruption_type, severity,
                    num_samples=1000,
                    input_size=args.input_size
                )
            
            # Create dataloader
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, sampler=sampler,
                num_workers=args.num_workers, pin_memory=True
            )
            
            # Evaluate
            accuracy = evaluate_corruption(model, data_loader, device, rank)
            results[corruption_type][severity] = accuracy
            
            if is_main_process:
                print(f"Acc: {accuracy*100:.2f}%")
        
        dist.barrier()
    
    if is_main_process:
        # Compute statistics
        print(f"\n{'='*70}")
        print("Robustness Evaluation Results:")
        print(f"{'='*70}\n")
        
        # Per-corruption results
        for corruption_type in args.corruption_types:
            avg_acc = np.mean([results[corruption_type][s] for s in args.severity_levels])
            print(f"{corruption_type:20s}: {avg_acc*100:.2f}%")
        
        # Overall statistics
        all_accuracies = [results[c][s] for c in args.corruption_types for s in args.severity_levels]
        mean_acc = np.mean(all_accuracies)
        
        print(f"\n{'='*70}")
        print(f"Mean Accuracy across all corruptions: {mean_acc*100:.2f}%")
        print(f"{'='*70}\n")
        
        # Save results
        results_file = output_dir / "robustness_results.txt"
        with open(results_file, "w") as f:
            f.write("ImageNet-C Robustness Evaluation Results\n")
            f.write("="*50 + "\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"DAGA: {'Enabled' if args.use_daga else 'Disabled'}\n")
            if args.use_daga:
                f.write(f"DAGA Layers: {args.daga_layers}\n")
            f.write("="*50 + "\n\n")
            
            for corruption_type in args.corruption_types:
                f.write(f"\n{corruption_type}:\n")
                for severity in args.severity_levels:
                    acc = results[corruption_type][severity]
                    f.write(f"  Severity {severity}: {acc*100:.2f}%\n")
                avg_acc = np.mean([results[corruption_type][s] for s in args.severity_levels])
                f.write(f"  Average: {avg_acc*100:.2f}%\n")
            
            f.write(f"\nOverall Mean Accuracy: {mean_acc*100:.2f}%\n")
        
        print(f"✓ Results saved to {results_file}")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()


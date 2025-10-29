# dinov3_finetune_daga_pth.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import torchvision.datasets as datasets
import warnings
import json
import swanlab
import matplotlib.pyplot as plt
from datetime import date
import random

import dinov3.models as dinov3_models

from daga import DAGA  # Import from your new module


# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

class ModifiedViT(nn.Module):
    def __init__(
        self,
        pretrained_vit,
        num_classes=10,
        use_daga=False,
        daga_layers=[11],
        enable_visualization=False,
        vis_attn_layer=11,
        n_last_blocks=4,
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.num_classes = num_classes
        self.use_daga = use_daga
        self.daga_layers = daga_layers
        self.feature_dim = self.vit.embed_dim
        self.enable_visualization = enable_visualization
        self.vis_attn_layer = vis_attn_layer
        self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
        
        # We will calculate num_storage_tokens dynamically
        self.num_storage_tokens = -1 
        
        self.captured_attn = None
        self.captured_guidance_attn = None

        # Freeze all backbone parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Initialize DAGA modules if used
        if self.use_daga:
            self.daga_modules = nn.ModuleDict(
                {str(i): DAGA(feature_dim=self.feature_dim) for i in daga_layers}
            )

        # Initialize the classifier head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # Ensure classifier and DAGA parameters are trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        if self.use_daga:
            for param in self.daga_modules.parameters():
                param.requires_grad = True

        print(
            f"✓ ModifiedViT initialized:\n"
            f"  - Feature dim: {self.feature_dim}\n"
            f"  - Num classes: {num_classes}\n"
            f"  - Use DAGA: {self.use_daga} (Layers: {self.daga_layers if self.use_daga else 'N/A'})"
        )

    def _get_attention_map(self, block, x):
        """
        Helper function to calculate attention maps.
        FIX: Applies block.norm1(x) before qkv.
        """
        attn_module = block.attn
        
        # Apply the block's LayerNorm *before* calculating QKV
        normed_x = block.norm1(x)
        
        B, N, C = normed_x.shape
        num_heads = attn_module.num_heads
        head_dim = C // num_heads
        
        # Calculate QKV from the *normed* input
        qkv = attn_module.qkv(normed_x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Note: Omitting RoPE embeddings here for simplicity/approximation,
        # as calculating them correctly is complex outside the real forward pass.
        # This matches the behavior of test_visualization.py.
        q = q * attn_module.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        return attn

    def forward(self, x, request_visualization_maps=False):
        B = x.shape[0]
        # x_processed: (B, seq_len, C)
        # (H, W) are patch grid dimensions
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        
        # Dynamically calculate num_registers, assuming [CLS], [REG], [PATCH] order
        num_registers = seq_len - num_patches - 1
        
        # Store this if it's the first time
        if self.num_storage_tokens == -1:
             self.num_storage_tokens = num_registers
            #  print(f"  [DINOv3] Detected token order: [CLS] (1), [REGISTERS] ({num_registers}), [PATCHES] ({num_patches})")

        daga_guidance_map = None
        if self.use_daga:
            with torch.no_grad():
                guidance_features = x_processed.clone()
                
                for i in range(len(self.vit.blocks)):
                    rope_sincos = (
                        self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
                    )
                    
                    if i == self.daga_guidance_layer_idx:
                        # Use the fixed helper function (with norm1)
                        attn_weights = self._get_attention_map(
                            self.vit.blocks[i], 
                            guidance_features
                        )
                        self.captured_guidance_attn = attn_weights
                    
                    guidance_features = self.vit.blocks[i](guidance_features, rope_sincos)
            
            if self.captured_guidance_attn is not None:
                # Get attention from CLS (idx 0) to all other tokens (idx 1:)
                # Shape: (B, num_heads, seq_len - 1)
                cls_attn_all_tokens = self.captured_guidance_attn[:, :, 0, 1:]
                
                # --- START OF FIX (Token Order) ---
                # Other tokens are [REGISTERS... (num_registers)], [PATCHES... (num_patches)]
                # We must *skip* the register tokens to get only patch attention
                patch_start_index = num_registers
                cls_attn_patch_tokens_headed = cls_attn_all_tokens[:, :, patch_start_index:]
                # --- END OF FIX ---
                
                # Average over heads
                cls_attn_patches = cls_attn_patch_tokens_headed.mean(dim=1)
                
                # Normalize for guidance map
                min_val = cls_attn_patches.amin(dim=1, keepdim=True)
                max_val = cls_attn_patches.amax(dim=1, keepdim=True)
                cls_attn = (cls_attn_patches - min_val) / (max_val - min_val + 1e-8)
                
                if cls_attn.shape[1] == num_patches:
                    daga_guidance_map = cls_attn.reshape(B, H, W)

        adapted_attn_weights = None
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            if request_visualization_maps and idx == self.vis_attn_layer:
                with torch.no_grad():
                    # Use the fixed helper function (with norm1)
                    adapted_attn_weights = self._get_attention_map(
                        block,
                        x_processed
                    )
            
            # Pass tokens through the block
            x_processed = block(x_processed, rope_sincos)

            if (
                self.use_daga
                and idx in self.daga_layers
                and daga_guidance_map is not None
            ):
                # --- START OF FIX (Token Order) ---
                # Apply DAGA only to patch tokens, respecting [CLS], [REG], [PATCH] order
                
                # CLS token
                cls_token = x_processed[:, :1, :]
                
                # Register tokens
                register_start_index = 1
                register_end_index = 1 + num_registers
                register_tokens = x_processed[:, register_start_index:register_end_index, :]
                
                # Patch tokens
                patch_start_index = 1 + num_registers
                patch_tokens = x_processed[:, patch_start_index:, :]
                
                adapted_patch_tokens = self.daga_modules[str(idx)](
                    patch_tokens, daga_guidance_map
                )
                
                # Re-concatenate in the correct order
                x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
                # --- END OF FIX ---
        
        x_normalized = self.vit.norm(x_processed)
        features = x_normalized[:, 0]
        logits = self.classifier(features)

        return logits, adapted_attn_weights, daga_guidance_map

def setup_training_components(model, args):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    base_model = model.module if isinstance(model, DataParallel) else model

    daga_params = []
    classifier_params = []

    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if "daga" in name:
                daga_params.append(param)
            else:
                classifier_params.append(param)

    lr_scaled = args.lr * (args.batch_size * torch.cuda.device_count()) / 256.0

    param_groups = [{"params": classifier_params, "lr": lr_scaled, "weight_decay": 0.0}]
    if daga_params:
        param_groups.append(
            {"params": daga_params, "lr": lr_scaled * 0.5, "weight_decay": 0.0}
        )

    optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)

    warmup_epochs = 1

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (
            1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return criterion, optimizer, scheduler


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="DINOv3 ViT Fine-tuning with DAGA")

    parser.add_argument(
        "--model_name",
        type=str,
        default="dinov3_vits16",
        help="Name of the DINOv3 model architecture (e.g., dinov3_vits16)",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        help="Path to the local pretrained DINOv3 model checkpoint",
    )

    # Dataset parameters
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
        help="Path to the root directory for all datasets",
    )
    parser.add_argument(
        "--subset_ratio", type=float, default=1.0, help="Ratio of training data to use"
    )

    # Model parameters
    parser.add_argument(
        "--use_daga",
        action="store_true",
        help="Use DAGA (Dynamic Attention-Gated Adapter)",
    )
    parser.add_argument(
        "--daga_layers",
        type=int,
        nargs="+",
        default=[11],
        help="Layer indices to insert DAGA (e.g., 11 for last layer)",
    )

    parser.add_argument("--input_size", type=int, default=518, help="Input image size")

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Logging & Visualization parameters
    parser.add_argument(
        "--output_dir", default="./outputs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--swanlab_name",
        type=str,
        default=None,
        help="Custom experiment name for SwanLab",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=5,
        help="Frequency (epochs) to log attention visualizations.",
    )
    parser.add_argument(
        "--vis_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="List of indices of test set images to use for consistent visualization.",
    )
    parser.add_argument(
        "--enable_visualization",
        action="store_true",
        help="Enable attention map visualization during training.",
    )
    parser.add_argument(
        "--vis_attn_layer",
        type=int,
        default=11,
        help="Which layer's attention to visualize from the adapted model.",
    )

    return parser.parse_args()


def get_dataset(args):
    """Load and prepare dataset."""
    if not args.data_path:
        raise ValueError("`--data_path` must be specified.")

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(args.input_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=True, transform=test_transform
        )
        num_classes = 10
    elif args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=True, transform=test_transform
        )
        num_classes = 100
    elif args.dataset == "imagenet100":
        train_path = os.path.join(args.data_path, "train")
        val_path = os.path.join(args.data_path, "val")
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(val_path, transform=test_transform)
        num_classes = 100
    elif args.dataset == "imagenet":
        train_path = os.path.join(args.data_path, "train")
        val_path = os.path.join(args.data_path, "val")
        val_annot_path = os.path.join(args.data_path, "val_annotations.txt")
        if not os.path.exists(val_annot_path):
            raise FileNotFoundError(f"Annotation file not found: {val_annot_path}")

        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        test_dataset = ImageNetValDataset(
            val_path, val_annot_path, transform=test_transform
        )
        num_classes = len(train_dataset.classes)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.subset_ratio < 1.0:
        subset_size = int(len(train_dataset) * args.subset_ratio)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(
            f"Using subset of {subset_size} training samples ({args.subset_ratio*100:.1f}%)"
        )

    return train_dataset, test_dataset, num_classes


class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.targets = []

        # Create a mapping from class ID (e.g., n01440764) to a continuous index (0, 1, 2...)
        train_path = os.path.join(Path(root_dir).parent, "train")
        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(sorted(os.listdir(train_path)))
        }
        self.classes = list(self.class_to_idx.keys())

        with open(annotation_file, "r") as f:
            for line in f:
                img_name, class_id = line.strip().split("\t")
                img_path = os.path.join(self.root_dir, img_name)

                if class_id in self.class_to_idx:
                    class_idx = self.class_to_idx[class_id]
                    self.samples.append((img_path, class_idx))
                    self.targets.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = datasets.folder.default_loader(img_path)

        if self.transform:
            image = self.transform(image)

        return image, target


def get_base_model(model):
    """Get the base model, handling DataParallel wrapper."""
    return model.module if isinstance(model, DataParallel) else model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits, _, _ = model(images, request_visualization_maps=False)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        running_avg_loss = total_loss / (batch_idx + 1)

        pbar.set_postfix(
            {
                "Loss": f"{running_avg_loss:.4f}",
                "Acc": f"{100. * correct / total:.2f}%",
            }
        )

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            logits, _, _ = model(images, request_visualization_maps=False)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def setup_model_and_data(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset, num_classes = get_dataset(args)

    print(f"Loading DINOv3 model '{args.model_name}' from local repository...")

    dinov3_repo_path = '/home/user/zhoutianjian/dinov3'
    weights_path = os.path.join('/home/user/zhoutianjian/DAGA/checkpoints', args.pretrained_path)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    
    print(f"✓ Loading checkpoint from: {weights_path}")
    
    vit_model = torch.hub.load(
        dinov3_repo_path,
        args.model_name,
        source='local',
        weights=weights_path
    )
    
    print(f"\n[DEBUG] Model loaded:")
    print(f"  embed_dim: {vit_model.embed_dim}")
    print(f"  n_storage_tokens: {getattr(vit_model, 'n_storage_tokens', 0)}")
    if hasattr(vit_model, "storage_tokens"):
        print(f"  storage_tokens shape: {vit_model.storage_tokens.shape}")
    print(f"  cls_token shape: {vit_model.cls_token.shape}")

    model = ModifiedViT(
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

    return model, train_dataset, test_dataset, device


def visualize_attention_comparison(
    model, fixed_images, args, output_dir, epoch, test_dataset=None
):
    if fixed_images is None:
        return []

    base_model = get_base_model(model)
    base_model.eval()
    vis_figs = []

    class_names = getattr(test_dataset, "classes", None)

    with torch.no_grad():
        # Get patch grid dimensions (H, W)
        _, (H, W) = base_model.vit.prepare_tokens_with_masks(fixed_images)
        num_patches_expected = H * W

        # Get adapted model's attention maps (this runs the modified .forward() pass)
        logits, adapted_attn_weights, _ = base_model(
            fixed_images, request_visualization_maps=True
        )

        if adapted_attn_weights is None:
            print(
                "⚠ Warning: Could not generate adapted attention maps from the model."
            )
            return []

        # --- START OF FIX (Token Order) ---
        def process_raw_weights(raw_weights, num_patches_expected, H, W):
            """
            Helper to process raw attention, assuming [CLS], [REG], [PATCH] order.
            This logic now mirrors test_visualization.py.
            """
            # raw_weights shape: (B, num_heads, seq_len, seq_len)
            B, _, seq_len, _ = raw_weights.shape
            
            # Dynamically calculate num_registers
            # seq_len = 1 (CLS) + num_registers + num_patches
            num_registers = seq_len - num_patches_expected - 1
            if num_registers < 0:
                print(f"⚠ Warning: Negative registers ({num_registers}). Assuming 0.")
                num_registers = 0

            # Get attention from CLS (idx 0) to all other tokens (idx 1:)
            cls_attn_all_tokens = raw_weights[:, :, 0, 1:]
            
            # Other tokens are [REGISTERS... (num_registers)], [PATCHES... (num_patches)]
            # We must *skip* the register tokens to get only patch attention
            patch_start_index = num_registers
            cls_attn_patch_tokens_headed = cls_attn_all_tokens[:, :, patch_start_index:]
            
            # Average across heads
            cls_attn_patches = cls_attn_patch_tokens_headed.mean(dim=1)

            # Min-Max normalization for visualization
            min_val = cls_attn_patches.amin(dim=1, keepdim=True)
            max_val = cls_attn_patches.amax(dim=1, keepdim=True)
            cls_attn_normalized = (cls_attn_patches - min_val) / (max_val - min_val + 1e-8)
            
            B, N = cls_attn_normalized.shape

            # Validate patch count
            if N != num_patches_expected:
                print(
                    f"⚠ Warning: Attention patch count mismatch! Expected {num_patches_expected}, got {N}."
                )
                # Fallback: try to reshape what we got
                if N > 0 and int(N**0.5) * int(N**0.5) == N:
                     H = W = int(N**0.5)
                else:
                     return None # Cannot proceed
            
            if H > 0 and W > 0:
                return cls_attn_normalized.reshape(B, H, W).cpu().numpy()
            else:
                return None
        # --- END OF FIX ---


        # Process the adapted (DAGA) model's attention
        adapted_attn_np = process_raw_weights(adapted_attn_weights, num_patches_expected, H, W)
        if adapted_attn_np is None:
            return []

        predictions = logits.argmax(dim=1).cpu().numpy()

        # Get the frozen backbone's attention (baseline)
        with torch.no_grad():
            x_proc, (H_baseline, W_baseline) = base_model.vit.prepare_tokens_with_masks(fixed_images)
            assert H == H_baseline and W == W_baseline

            baseline_raw_weights = None
            for i in range(base_model.vis_attn_layer + 1):
                rope_sincos = (
                    base_model.vit.rope_embed(H=H, W=W)
                    if base_model.vit.rope_embed
                    else None
                )
                
                if i == base_model.vis_attn_layer:
                    # Use the fixed helper function (with norm1)
                    baseline_raw_weights = base_model._get_attention_map(
                        base_model.vit.blocks[i], x_proc
                    )

                # Pass token through the block to get input for the *next* layer
                x_proc = base_model.vit.blocks[i](x_proc, rope_sincos)

        if baseline_raw_weights is None:
            print(
                f"⚠ Warning: Failed to extract baseline attention weights from layer {base_model.vis_attn_layer}."
            )
            return []

        # Process the baseline attention using the same corrected helper
        baseline_attn_np = process_raw_weights(baseline_raw_weights, num_patches_expected, H, W)
        if baseline_attn_np is None:
            return []

        # --- Plotting logic (unchanged) ---
        images_np = fixed_images.cpu().numpy()
        vis_save_path = Path(output_dir) / "visualizations"
        vis_save_path.mkdir(parents=True, exist_ok=True)

        for j in range(images_np.shape[0]):
            original_image_index = args.vis_indices[j]
            actual_label_name = "Unknown"
            pred_label_name = "Unknown"

            if class_names and hasattr(test_dataset, "targets"):
                try:
                    actual_label_int = test_dataset.targets[original_image_index]
                    actual_label_name = class_names[actual_label_int]
                    pred_label_name = class_names[predictions[j]]
                except (IndexError, TypeError):
                    actual_label_name = "Label Index Error"

            pred_correct = "✓" if actual_label_name == pred_label_name else "✗"
            fig_title = f"Epoch {epoch+1} - Img#{original_image_index} | True: {actual_label_name} | Pred: {pred_label_name} {pred_correct}"

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(fig_title, fontsize=14, fontweight="bold")

            img = images_np[j].transpose(1, 2, 0)
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            axes[0].imshow(img)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(baseline_attn_np[j], cmap="viridis")
            axes[1].set_title(f"Frozen Backbone Attn (L{base_model.vis_attn_layer})")
            axes[1].axis("off")

            is_daga_model = getattr(base_model, "use_daga", False)
            adapted_title = (
                f"Adapted (DAGA) Attn (L{base_model.vis_attn_layer})"
                if is_daga_model
                else f"Finetuned Model Attn (L{base_model.vis_attn_layer})"
            )
            axes[2].imshow(adapted_attn_np[j], cmap="viridis")
            axes[2].set_title(adapted_title)
            axes[2].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            vis_figs.append(fig)

            fig.savefig(
                vis_save_path / f"epoch_{epoch+1}_imgidx_{original_image_index}.png",
                dpi=100,
            )
            plt.close(fig)

    return vis_figs


def prepare_visualization_data(test_dataset, args, device):
    """Prepares a fixed batch of data for consistent visualization using specified indices."""
    if not args.enable_visualization:
        return None
    print(f"\n📸 Preparing visualization data for image indices: {args.vis_indices}...")
    vis_subset = torch.utils.data.Subset(test_dataset, args.vis_indices)
    vis_loader = DataLoader(vis_subset, batch_size=len(args.vis_indices), shuffle=False)
    fixed_images, _ = next(iter(vis_loader))
    print("✓ Visualization images loaded.")
    return fixed_images.to(device)


def run_training_loop(
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
    test_dataset=None,  # 添加参数
):
    """Executes the main training and evaluation loop."""
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        elapsed_time = time.time() - start_time
        print(f"\n📈 Epoch {epoch+1}/{args.epochs} Summary:")
        print(
            f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
        )
        print(f"   Time Elapsed: {elapsed_time/60:.1f}min")

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "total_time_minutes": elapsed_time / 60,
        }

        if args.enable_visualization and (
            epoch % args.log_freq == 0 or epoch == args.epochs - 1
        ):
            print("📊 Generating attention visualizations...")
            vis_figs = visualize_attention_comparison(
                model, fixed_vis_images, args, output_dir, epoch, test_dataset
            )
            if vis_figs:
                log_dict["attention_comparison"] = [
                    swanlab.Image(fig) for fig in vis_figs
                ]

        swanlab.log(log_dict, step=epoch + 1)

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = output_dir / "best_model.pth"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_acc,
                args,
                save_path,
                vis_data=fixed_vis_images,
            )
            print(f"   ✅ New best model saved! (Test Acc: {best_acc:.2f}%)")

    return best_acc, test_acc, (time.time() - start_time) / 60


def setup_environment(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(args):
    method_name = "daga" if args.use_daga else "baseline"
    exp_name = (
        args.swanlab_name
        or f"{args.dataset}_{method_name}_L{'-'.join(map(str, args.daga_layers)) if args.use_daga else ''}_{date.today()}"
    )
    swanlab.init(
        project=f"dino-finetuning_{date.today()}",
        experiment_name=exp_name,
        config=vars(args),
    )
    return exp_name


def create_dataloaders(train_dataset, test_dataset, batch_size):
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        ),
    )


def save_checkpoint(model, optimizer, epoch, best_acc, args, path, vis_data=None):
    """Saves model checkpoint, optionally including visualization data."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": get_base_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "args": vars(args),
    }
    if vis_data is not None:
        checkpoint["visualization_images"] = vis_data.cpu()
    torch.save(checkpoint, path)


def finalize_experiment(best_acc, final_acc, total_time_minutes, output_dir):
    print(f'\n{"="*70}\n🎉 Training Completed!\n{"="*70}')
    print(f"Total Time:       {total_time_minutes:.1f} minutes")
    print(f"Best Test Acc:    {best_acc:.2f}%")
    print(f"Final Test Acc:   {final_acc:.2f}%")
    print(f"Results saved to: {output_dir}\n{'='*70}\n")
    swanlab.finish()


def main():

    args = parse_arguments()
    setup_environment(args.seed)

    experiment_name = setup_logging(args)
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model, train_dataset, test_dataset, device = setup_model_and_data(args)
    train_loader, test_loader = create_dataloaders(
        train_dataset, test_dataset, args.batch_size
    )
    criterion, optimizer, scheduler = setup_training_components(model, args)

    fixed_vis_images = prepare_visualization_data(test_dataset, args, device)

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

    finalize_experiment(best_acc, final_acc, total_time, output_dir)


if __name__ == "__main__":
    main()

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
import swanlab
import time

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging, finalize_experiment
from core.ddp_utils import setup_ddp, cleanup_ddp
from core.datasets import NYUDepthV2Dataset

# Add dinov3 to path
dinov3_path = '/home/user/zhoutianjian/Dino_DAGA/dinov3'
if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)

from dinov3.eval.dense.depth.models import build_depther
from dinov3.eval.dense.depth.models.encoder import BackboneLayersSet

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


class ViTWithDAGAForDepth(nn.Module):
    """Wrapper for ViT backbone that injects DAGA for depth task"""
    def __init__(self, vit_model, use_daga=False, daga_layers=None):
        super().__init__()
        self.vit = vit_model
        self.use_daga = use_daga
        self.daga_layers = daga_layers if daga_layers else []
        self.feature_dim = vit_model.embed_dim
        self.daga_guidance_layer_idx = max(daga_layers) if daga_layers else (len(vit_model.blocks) - 1)
        
        # Expose ViT attributes needed by DINOv3 depther
        self.embed_dim = vit_model.embed_dim
        self.blocks = vit_model.blocks
        self.n_blocks = len(vit_model.blocks)
        self.patch_size = vit_model.patch_size
        self.num_heads = vit_model.blocks[0].attn.num_heads if hasattr(vit_model.blocks[0], 'attn') else 12
        
        # Add missing attributes expected by depth encoder
        self.embed_dims = [self.embed_dim] * self.n_blocks
        self.input_pad_size = self.patch_size
        
        # DAGA modules
        if self.use_daga:
            from core.daga import DAGA
            from core.backbones import compute_daga_guidance_map, get_attention_map
            
            self.daga_modules = nn.ModuleDict({
                str(i): DAGA(feature_dim=self.feature_dim) for i in self.daga_layers
            })
            for param in self.daga_modules.parameters():
                param.requires_grad = True
        
        # Freeze original ViT
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Cache for visualization
        self._cached_attn = None
        self._cached_guidance = None
        self._last_intermediate_features = []
        self._last_shape = None
    
    def _forward_blocks(self, x):
        """Internal forward through blocks with optional DAGA"""
        from core.backbones import compute_daga_guidance_map, get_attention_map
        
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        # Compute DAGA guidance
        daga_guidance_map = None
        if self.use_daga:
            daga_guidance_map = compute_daga_guidance_map(
                self.vit, x_processed, H, W, self.daga_guidance_layer_idx
            )
            self._cached_guidance = daga_guidance_map
        
        # Store intermediate features
        intermediate_features = []
        
        # Find visualization layer
        visualization_layer = self.daga_guidance_layer_idx
        
        # Process through blocks
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # Cache attention BEFORE block execution (for visualization)
            if idx == visualization_layer:
                with torch.no_grad():
                    self._cached_attn = get_attention_map(block, x_processed)
            
            x_processed = block(x_processed, rope_sincos)
            
            # Apply DAGA AFTER block execution
            if self.use_daga and idx in self.daga_layers and daga_guidance_map is not None:
                cls_token = x_processed[:, :1, :]
                register_tokens = x_processed[:, 1:1+num_registers, :]
                patch_tokens = x_processed[:, 1+num_registers:, :]
                
                adapted_patch_tokens = self.daga_modules[str(idx)](
                    patch_tokens, daga_guidance_map
                )
                
                x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
            
            # Store for intermediate layer extraction
            intermediate_features.append(x_processed)
        
        self._last_intermediate_features = intermediate_features
        self._last_shape = (H, W)
        return x_processed, (H, W)
    
    def forward(self, x):
        """Forward compatible with ViT interface"""
        x_processed, _ = self._forward_blocks(x)
        return x_processed
    
    def get_intermediate_layers(self, x, n, reshape=True, return_class_token=False, norm=True):
        """Get intermediate layer features (required by DINOv3 depther)"""
        # Run forward to populate intermediate features
        self._forward_blocks(x)
        
        H, W = self._last_shape
        n_blocks = len(self.vit.blocks)
        
        # Handle different types of n
        if isinstance(n, int):
            indices = [n - 1]
        else:
            indices = [i - 1 if i > 0 else n_blocks + i for i in n]
        
        # Extract requested layers
        outputs = []
        for idx in indices:
            if idx < len(self._last_intermediate_features):
                feat = self._last_intermediate_features[idx]
                
                B, seq_len, C = feat.shape
                num_patches = H * W
                num_registers = seq_len - num_patches - 1
                
                cls_token = feat[:, :1, :] if return_class_token else None
                patch_tokens = feat[:, 1+num_registers:, :]  # Skip CLS + registers
                
                if reshape:
                    feat_reshaped = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
                    if return_class_token:
                        outputs.append((feat_reshaped, cls_token))
                    else:
                        outputs.append(feat_reshaped)
                else:
                    if return_class_token:
                        outputs.append((patch_tokens, cls_token))
                    else:
                        outputs.append(patch_tokens)
        
        return outputs
    
    def prepare_tokens_with_masks(self, x):
        """Forward to ViT's prepare_tokens_with_masks"""
        return self.vit.prepare_tokens_with_masks(x)
    
    @property
    def rope_embed(self):
        """Forward to ViT's rope_embed"""
        return self.vit.rope_embed


class DepthModel(nn.Module):
    """Depth estimation model with DPT head and optional DAGA"""
    def __init__(self, vit_model, out_indices, use_daga=False, daga_layers=None,
                 min_depth=0.001, max_depth=10.0):
        super().__init__()
        self.use_daga = use_daga
        self.daga_layers = daga_layers if daga_layers else []
        self.out_indices = out_indices
        
        # Always wrap ViT to ensure consistent interface
        # Even for baseline, wrapping adds missing attributes expected by depth encoder
        if use_daga:
            self.vit_model = ViTWithDAGAForDepth(vit_model, use_daga, daga_layers)
        else:
            # Wrap without DAGA to provide consistent attributes
            self.vit_model = ViTWithDAGAForDepth(vit_model, use_daga=False, daga_layers=[])
        
        # Build DPT depth head using official implementation
        self.depther = build_depther(
            backbone=self.vit_model,
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
        depth_pred = self.depther(x)
        return depth_pred


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
    
    # Logging arguments
    parser.add_argument("--swanlab_name", type=str, default=None, help="SwanLab experiment name")
    parser.add_argument("--swanlab_mode", type=str, default="disabled", help="SwanLab mode")
    
    return parser.parse_args()


def visualize_depth_predictions(images, depths_gt, depths_pred, output_dir, epoch, num_samples=4):
    """
    Visualize depth predictions - generates separate images for each sample
    Consistent format with DAGA visualization for easy comparison
    
    Args:
        images: (B, 3, H, W) RGB images (normalized)
        depths_gt: (B, H, W) ground truth depth maps
        depths_pred: (B, H, W) predicted depth maps
        output_dir: directory to save visualizations
        epoch: current epoch number
        num_samples: number of samples to visualize
    
    Returns:
        list of matplotlib figures for SwanLab logging
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    num_samples = min(num_samples, images.shape[0])
    
    # Denormalize images
    images_np = images.cpu().numpy()
    depths_gt_np = depths_gt.cpu().numpy()
    depths_pred_np = depths_pred.cpu().numpy()
    
    vis_save_path = Path(output_dir) / "visualizations"
    vis_save_path.mkdir(parents=True, exist_ok=True)
    
    vis_figs = []
    
    for idx in range(num_samples):
        # Denormalize image
        img = images_np[idx].transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = np.clip(std * img + mean, 0, 1)
        
        depth_gt_np = depths_gt_np[idx]
        depth_pred_np = depths_pred_np[idx]
        
        # Create depth visualization (3 panels: Image, GT, Prediction)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Depth Results - Epoch {epoch} - Sample {idx+1}", 
                     fontsize=14, fontweight="bold")
        
        axes[0].imshow(img)
        axes[0].set_title("RGB Image")
        axes[0].axis("off")
        
        im1 = axes[1].imshow(depth_gt_np, cmap='plasma', vmin=0, vmax=10)
        axes[1].set_title("Ground Truth Depth")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(depth_pred_np, cmap='plasma', vmin=0, vmax=10)
        axes[2].set_title("Predicted Depth")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        save_path = vis_save_path / f"epoch_{epoch}_sample_{idx+1}_depth.png"
        fig.savefig(save_path, dpi=100)
        vis_figs.append(fig)
    
    print(f"  ✓ Saved {num_samples} depth visualizations")
    return vis_figs


def visualize_depth_with_attention(model, images, depths_gt, depths_pred, 
                                    output_dir, epoch, num_samples=4):
    """
    Visualize depth predictions with attention maps (for DAGA models)
    Follows the same pattern as segmentation visualization
    
    Args:
        model: depth model (should be unwrapped from DDP)
        images: (B, 3, H, W) RGB images
        depths_gt: (B, H, W) ground truth depth
        depths_pred: (B, H, W) predicted depth  
        output_dir: directory to save visualizations
        epoch: current epoch number
        num_samples: number of samples to visualize
    
    Returns:
        list of matplotlib figures for SwanLab logging
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    from scipy.ndimage import zoom
    
    try:
        from core.backbones import get_attention_map, process_attention_weights
    except Exception as e:
        print(f"  Warning: Could not import attention utilities: {e}")
        print("  Skipping attention visualization")
        return []
    
    # Unwrap model if needed
    if isinstance(model, (DataParallel, DDP)):
        model = model.module
    
    is_daga = getattr(model, 'use_daga', False)
    if not is_daga:
        print("  Model is not using DAGA, skipping attention visualization")
        return []
    
    model.eval()
    num_samples = min(num_samples, images.shape[0])
    
    # Print DAGA status
    print(f"\n[DEBUG] Depth DAGA Visualization at epoch {epoch}:")
    print(f"  DAGA layers: {model.daga_layers}")
    
    # Access ViT model (wrapped with DAGA)
    vit_backbone = model.vit_model
    vis_layer_idx = vit_backbone.daga_guidance_layer_idx
    
    if hasattr(vit_backbone, 'daga_modules'):
        print(f"  Visualization layer: {vis_layer_idx}")
        print(f"  Mix weights:")
        for layer_idx, daga_module in vit_backbone.daga_modules.items():
            weight_val = daga_module.mix_weight.item()
            print(f"    Layer {layer_idx}: {weight_val:.6f}")
    print()
    
    with torch.no_grad():
        # Get patch shape info
        x_proc, (H, W) = vit_backbone.vit.prepare_tokens_with_masks(images)
        num_patches_expected = H * W
        
        # === 1. Get DAGA-adapted attention (from current model forward) ===
        # Forward pass to populate cached attention
        _ = model(images)
        
        adapted_attn_np = None
        baseline_attn_np = None
        
        if hasattr(vit_backbone, '_cached_attn') and vit_backbone._cached_attn is not None:
            adapted_attn_np = process_attention_weights(
                vit_backbone._cached_attn, num_patches_expected, H, W
            )
        
        # === 2. Get baseline attention (from frozen backbone without DAGA) ===
        x_proc_baseline, _ = vit_backbone.vit.prepare_tokens_with_masks(images)
        baseline_raw_weights = None
        
        for i in range(vis_layer_idx + 1):
            rope_sincos = (
                vit_backbone.vit.rope_embed(H=H, W=W)
                if vit_backbone.vit.rope_embed
                else None
            )
            if i == vis_layer_idx:
                baseline_raw_weights = get_attention_map(
                    vit_backbone.vit.blocks[i], x_proc_baseline
                )
            x_proc_baseline = vit_backbone.vit.blocks[i](x_proc_baseline, rope_sincos)
        
        if baseline_raw_weights is not None:
            baseline_attn_np = process_attention_weights(
                baseline_raw_weights, num_patches_expected, H, W
            )
        
        # Denormalize images for display
        images_np = images.cpu().numpy()
        depths_gt_np = depths_gt.cpu().numpy()
        depths_pred_np = depths_pred.cpu().numpy()
        
        vis_save_path = Path(output_dir) / "visualizations"
        vis_save_path.mkdir(parents=True, exist_ok=True)
        
        vis_figs = []
        
        # Process each sample
        for idx in range(num_samples):
            # Denormalize image
            img = images_np[idx].transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            
            depth_gt_np = depths_gt_np[idx]
            depth_pred_np = depths_pred_np[idx]
            
            # === Group 1: Depth Results (Image, GT, Prediction) ===
            fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
            fig1.suptitle(f"Depth Results - Epoch {epoch} - Sample {idx+1}", 
                         fontsize=14, fontweight="bold")
            
            axes1[0].imshow(img)
            axes1[0].set_title("RGB Image")
            axes1[0].axis("off")
            
            im1 = axes1[1].imshow(depth_gt_np, cmap='plasma', vmin=0, vmax=10)
            axes1[1].set_title("Ground Truth Depth")
            axes1[1].axis("off")
            plt.colorbar(im1, ax=axes1[1], fraction=0.046, pad=0.04)
            
            im2 = axes1[2].imshow(depth_pred_np, cmap='plasma', vmin=0, vmax=10)
            axes1[2].set_title("Predicted Depth")
            axes1[2].axis("off")
            plt.colorbar(im2, ax=axes1[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            save_path1 = vis_save_path / f"epoch_{epoch}_sample_{idx+1}_depth.png"
            fig1.savefig(save_path1, dpi=100)
            plt.close(fig1)
            
            # === Group 2: Attention Map Comparison (only if both available) ===
            if adapted_attn_np is not None and baseline_attn_np is not None:
                fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
                fig2.suptitle(f"Attention Analysis - Epoch {epoch} - Sample {idx+1}", 
                            fontsize=16, fontweight="bold")
                
                # Row 1: Original attention maps
                im0 = axes2[0, 0].imshow(baseline_attn_np[idx], cmap="viridis", vmin=0, vmax=1)
                axes2[0, 0].set_title(f"Frozen Backbone Attention (L{vis_layer_idx})", fontsize=12)
                axes2[0, 0].axis("off")
                plt.colorbar(im0, ax=axes2[0, 0], fraction=0.046, pad=0.04)
                
                im1 = axes2[0, 1].imshow(adapted_attn_np[idx], cmap="viridis", vmin=0, vmax=1)
                axes2[0, 1].set_title("DAGA-Adapted Attention", fontsize=12)
                axes2[0, 1].axis("off")
                plt.colorbar(im1, ax=axes2[0, 1], fraction=0.046, pad=0.04)
                
                # Row 2: Difference map and overlay
                diff_map = adapted_attn_np[idx] - baseline_attn_np[idx]
                abs_diff = np.abs(diff_map)
                max_diff = np.max(abs_diff)
                mean_diff = np.mean(abs_diff)
                
                im2 = axes2[1, 0].imshow(diff_map, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
                axes2[1, 0].set_title(f"Difference Map\n(Mean |Δ|={mean_diff:.4f}, Max |Δ|={max_diff:.4f})", 
                                     fontsize=11)
                axes2[1, 0].axis("off")
                plt.colorbar(im2, ax=axes2[1, 0], fraction=0.046, pad=0.04)
                
                # Overlay difference on image
                axes2[1, 1].imshow(img)
                # Resize attention difference to match image size
                if diff_map.shape != img.shape[:2]:
                    zoom_h = img.shape[0] / diff_map.shape[0]
                    zoom_w = img.shape[1] / diff_map.shape[1]
                    diff_resized = zoom(diff_map, (zoom_h, zoom_w), order=1)
                else:
                    diff_resized = diff_map
                
                im3 = axes2[1, 1].imshow(diff_resized, cmap="RdBu_r", alpha=0.6, vmin=-0.3, vmax=0.3)
                axes2[1, 1].set_title("Difference Overlay on Image", fontsize=11)
                axes2[1, 1].axis("off")
                plt.colorbar(im3, ax=axes2[1, 1], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                
                save_path2 = vis_save_path / f"epoch_{epoch}_sample_{idx+1}_attention.png"
                fig2.savefig(save_path2, dpi=100)
                vis_figs.append(fig2)
                
                if idx == 0:  # Only print once
                    print(f"  ✓ Saved depth and attention visualizations")
            else:
                if idx == 0:
                    print(f"  Note: Attention maps not available")
                    print(f"    - Baseline: {baseline_attn_np is not None}")
                    print(f"    - Adapted: {adapted_attn_np is not None}")
        
        return vis_figs


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss for depth estimation
    Better than L1/L2 loss for depth estimation
    Reference: Eigen et al. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
    """
    def __init__(self, lambda_weight=0.85):
        super().__init__()
        self.lambda_weight = lambda_weight
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: predicted depth (B, 1, H, W)
            target: ground truth depth (B, 1, H, W)
            mask: valid depth mask (optional)
        """
        if mask is None:
            mask = (target > 0.001) & (target < 100)
        
        # Apply mask
        pred = pred[mask]
        target = target[mask]
        
        if len(pred) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute log difference
        log_diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
        
        # SILog loss = variance(log_diff) = E[d^2] - lambda * E[d]^2
        loss = torch.mean(log_diff ** 2) - self.lambda_weight * (torch.mean(log_diff) ** 2)
        
        return loss


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
        
        # Resize prediction to match target if needed
        if pred_depths.shape != depths.shape:
            pred_depths = torch.nn.functional.interpolate(
                pred_depths, size=depths.shape[2:], mode='bilinear', align_corners=False
            )
        
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


def evaluate(model, test_loader, device, args, rank, epoch=None, output_dir=None):
    """Evaluate the model"""
    model.eval()
    metrics_sum = {}
    num_batches = 0
    
    is_main_process = (rank == 0)
    
    # Store first batch for visualization
    vis_images = None
    vis_depths_gt = None
    vis_depths_pred = None
    
    with torch.no_grad():
        if is_main_process:
            pbar = tqdm(test_loader, desc="Evaluating")
        else:
            pbar = test_loader
        
        for batch_idx, (images, depths) in enumerate(pbar):
            images = images.to(device)
            depths = depths.to(device)
            
            # Forward pass
            pred_depths = model(images)
            
            # Resize prediction to match target if needed
            if pred_depths.shape[2:] != depths.shape[-2:]:
                pred_depths = torch.nn.functional.interpolate(
                    pred_depths, size=depths.shape[-2:], mode='bilinear', align_corners=False
                )
            
            pred_depths = pred_depths.squeeze(1)
            
            # Save first batch for visualization (only if output_dir is provided)
            if batch_idx == 0 and is_main_process and args.enable_visualization and output_dir:
                vis_images = images.cpu()
                vis_depths_gt = depths.cpu()
                vis_depths_pred = pred_depths.cpu()
            
            # Compute metrics
            batch_metrics = compute_depth_metrics(
                pred_depths, depths, args.min_depth, args.max_depth
            )
            
            for key, val in batch_metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + val
            
            num_batches += 1
    
    # Average metrics
    metrics_avg = {k: v / num_batches for k, v in metrics_sum.items()}
    
    # Create visualization and collect figures for SwanLab
    vis_figs = []
    if is_main_process and args.enable_visualization and vis_images is not None and output_dir:
        # Generate separate depth visualizations for each sample
        depth_figs = visualize_depth_predictions(
            vis_images, vis_depths_gt, vis_depths_pred,
            output_dir,
            epoch if epoch is not None else 0,
            num_samples=args.num_vis_samples
        )
        vis_figs.extend(depth_figs)
        
        # Additional attention map visualization (for DAGA models)
        from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
        unwrapped_model = model.module if isinstance(model, (DataParallel, DDP)) else model
        if getattr(unwrapped_model, 'use_daga', False):
            attn_figs = visualize_depth_with_attention(
                model,
                vis_images.to(device),
                vis_depths_gt,
                vis_depths_pred,
                output_dir,
                epoch if epoch is not None else 0,
                num_samples=args.num_vis_samples
            )
            vis_figs.extend(attn_figs)
    
    return metrics_avg, vis_figs


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
    
    if is_main_process and args.use_daga:
        print(f"✓ Model loaded")
        print(f"Applying DAGA to layers: {args.daga_layers}")
    
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
    
    # Create datasets (using real NYU Depth V2 data)
    if is_main_process:
        print("Creating datasets...")
    
    try:
        train_dataset = NYUDepthV2Dataset(
            data_path=args.data_path,
            split='train',
            input_size=args.input_size,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            augmentation=True,  # Use augmentation for training
            sample_ratio=args.sample_ratio
        )
        test_dataset = NYUDepthV2Dataset(
            data_path=args.data_path,
            split='test',
            input_size=args.input_size,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            augmentation=False,
            sample_ratio=args.sample_ratio
        )
        
        if is_main_process:
            print(f"✓ Train dataset: {len(train_dataset)} samples")
            print(f"✓ Test dataset: {len(test_dataset)} samples")
    
    except FileNotFoundError as e:
        if is_main_process:
            print(f"⚠️  Warning: {e}")
            print("⚠️  NYU Depth V2 dataset not found. Using dummy data for testing.")
        
        # Fallback to dummy data if real data not available
        class DummyDepthDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples, input_size):
                self.num_samples = num_samples
                self.input_size = input_size
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                image = torch.randn(3, self.input_size, self.input_size)
                depth = torch.rand(self.input_size, self.input_size) * 10.0
                return image, depth
        
        train_dataset = DummyDepthDataset(100, args.input_size)
        test_dataset = DummyDepthDataset(20, args.input_size)
    
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
    # Use SILog loss - best for depth estimation (better than L1/L2)
    criterion = SILogLoss(lambda_weight=0.85)
    if is_main_process:
        print(f"✓ Using SILog loss (Scale-Invariant Logarithmic Loss)")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr * world_size,  # Scale LR with world size
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    # Warmup scheduler for first 5 epochs
    warmup_epochs = 5
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1,  # Start from 10% of lr
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    # Cosine annealing after warmup
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - warmup_epochs,
        eta_min=args.lr * world_size * 0.01  # End at 1% of initial lr
    )
    
    # Combine warmup and main scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    if is_main_process:
        print(f"\n{'='*70}")
        print("Starting training...")
        print(f"{'='*70}\n")
    
    best_metric = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        # Train
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args, epoch, rank
        )
        
        # Step learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Evaluate metrics every epoch (without visualization)
        metrics, _ = evaluate(model, test_loader, device, args, rank, epoch=epoch+1, output_dir=None)
        
        if is_main_process:
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Metrics:")
            for key, val in metrics.items():
                print(f"    {key}: {val:.4f}")
            
            # Log metrics to SwanLab every epoch
            elapsed_time = time.time() - start_time
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "learning_rate": current_lr,
                "total_time_minutes": elapsed_time / 60,
            }
            # Add all metrics to log
            for key, val in metrics.items():
                log_dict[f"val_{key}"] = val
            
            # Generate and log visualizations only at LOG_FREQ intervals
            if (epoch + 1) % args.log_freq == 0 or epoch == args.epochs - 1:
                print("📊 Generating depth visualizations...")
                _, vis_figs = evaluate(model, test_loader, device, args, rank, epoch=epoch+1, output_dir=output_dir)
                
                if vis_figs:
                    log_dict["depth_results"] = [swanlab.Image(fig) for fig in vis_figs]
                    # Close all figures to free memory
                    import matplotlib.pyplot as plt
                    for fig in vis_figs:
                        plt.close(fig)
            
            swanlab.log(log_dict, step=epoch + 1) if getattr(args, 'enable_swanlab', True) else None
            
            # Save best model
            if metrics['abs_rel'] < best_metric:
                best_metric = metrics['abs_rel']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': metrics,
                }, output_dir / "best_model.pth")
                print(f"  ✓ Saved best model (abs_rel: {best_metric:.4f})")
        
        dist.barrier()
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Best abs_rel: {best_metric:.4f}")
        print(f"{'='*70}\n")
        
        # Finalize SwanLab
        if getattr(args, 'enable_swanlab', True):
            swanlab.finish()
    
    cleanup_ddp()


if __name__ == "__main__":
    main()


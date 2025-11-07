"""
Model Comparison Visualization Tool
Compares baseline and DAGA models across different tasks (Classification, Detection, Segmentation)
Identifies and visualizes samples where DAGA shows the largest improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.backbones import load_dinov3_backbone
from core.backbones import process_attention_weights as process_attn_core
from tasks.classification import ClassificationModel
from tasks.segmentation import SegmentationModel, calculate_miou
from data.classification_datasets import get_classification_dataset
from data.segmentation_datasets import get_segmentation_dataset


class ModelComparer:
    """Compare baseline and DAGA models for different tasks"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup for multi-GPU
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            print(f"🔥 Using {self.gpu_count} GPUs for inference")
        
    def load_checkpoint_args(self, checkpoint_path):
        """Load args from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "args" in checkpoint:
            return argparse.Namespace(**checkpoint["args"])
        return None
        
    def load_classification_model(self, checkpoint_path, num_classes):
        """Load classification model from checkpoint"""
        print(f"\n📁 Loading classification model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_args = self.load_checkpoint_args(checkpoint_path)
        
        # Get model architecture from checkpoint, fallback to CLI args
        model_name = ckpt_args.model_name if ckpt_args and hasattr(ckpt_args, 'model_name') else self.args.model_name
        pretrained_path = ckpt_args.pretrained_path if ckpt_args and hasattr(ckpt_args, 'pretrained_path') else self.args.pretrained_path
        
        print(f"  Using model architecture: {model_name}")
        
        # Load backbone
        vit_model = load_dinov3_backbone(model_name, pretrained_path)
        
        # Create model
        use_daga = ckpt_args.use_daga if ckpt_args else False
        daga_layers = ckpt_args.daga_layers if ckpt_args else []
        
        model = ClassificationModel(
            vit_model,
            num_classes=num_classes,
            use_daga=use_daga,
            daga_layers=daga_layers
        )
        
        # Handle DataParallel wrapped models
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        # Use DataParallel for multi-GPU
        if self.gpu_count > 1:
            model = nn.DataParallel(model)
            print(f"  Wrapped with DataParallel ({self.gpu_count} GPUs)")
        
        model.eval()
        
        print(f"✓ Model loaded (DAGA: {use_daga}, Layers: {daga_layers})")
        return model
        
    def load_detection_model(self, checkpoint_path, num_classes):
        """Load detection model from checkpoint (not implemented)"""
        raise NotImplementedError("Detection model comparison not yet implemented")
        
    def load_segmentation_model(self, checkpoint_path, num_classes):
        """Load segmentation model from checkpoint"""
        print(f"\n📁 Loading segmentation model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_args = self.load_checkpoint_args(checkpoint_path)
        
        # Get model architecture from checkpoint, fallback to CLI args
        model_name = ckpt_args.model_name if ckpt_args and hasattr(ckpt_args, 'model_name') else self.args.model_name
        pretrained_path = ckpt_args.pretrained_path if ckpt_args and hasattr(ckpt_args, 'pretrained_path') else self.args.pretrained_path
        
        print(f"  Using model architecture: {model_name}")
        
        # Load backbone
        vit_model = load_dinov3_backbone(model_name, pretrained_path)
        
        # Create model
        use_daga = ckpt_args.use_daga if ckpt_args else False
        daga_layers = ckpt_args.daga_layers if ckpt_args else []
        out_indices = ckpt_args.out_indices if ckpt_args else [2, 5, 8, 11]
        
        model = SegmentationModel(
            vit_model,
            num_classes=num_classes,
            use_daga=use_daga,
            daga_layers=daga_layers,
            out_indices=out_indices
        )
        
        # Handle DataParallel wrapped models
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        # Use DataParallel for multi-GPU
        if self.gpu_count > 1:
            model = nn.DataParallel(model)
            print(f"  Wrapped with DataParallel ({self.gpu_count} GPUs)")
        
        model.eval()
        
        print(f"✓ Model loaded (DAGA: {use_daga}, Layers: {daga_layers})")
        return model
        
    def _process_attention(self, attn_weights, idx, H, W):
        """Process attention weights to create heatmap for a single sample"""
        if attn_weights is None:
            return None
        
        try:
            # Use the same processing as original code
            # attn_weights shape: (B, num_heads, num_tokens, num_tokens)
            num_patches_expected = H * W
            
            # Process single sample
            single_attn = attn_weights[idx:idx+1]  # Keep batch dimension
            attn_map_batch = process_attn_core(single_attn, num_patches_expected, H, W)
            
            if attn_map_batch is not None and len(attn_map_batch) > 0:
                return attn_map_batch[0]  # Return first (and only) sample
            else:
                return None
                
        except Exception as e:
            print(f"Warning: Could not process attention: {e}")
            return None
        
    def compare_classification(self):
        """Compare classification models"""
        print("\n" + "="*70)
        print("🎯 Classification Comparison")
        print("="*70)
        
        # Load dataset
        _, val_dataset, num_classes = get_classification_dataset(self.args)
        
        # Limit dataset for faster comparison
        if len(val_dataset) > self.args.max_samples:
            indices = list(range(self.args.max_samples))
            val_dataset = Subset(val_dataset, indices)
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Load models
        model_baseline = self.load_classification_model(
            self.args.baseline_path, num_classes
        )
        model_daga = self.load_classification_model(
            self.args.daga_path, num_classes
        )
        
        # Get actual models for H, W extraction
        from torch.nn.parallel import DataParallel
        actual_baseline = model_baseline.module if isinstance(model_baseline, DataParallel) else model_baseline
        actual_daga = model_daga.module if isinstance(model_daga, DataParallel) else model_daga
        
        # Find differences
        improvements = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Comparing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get H, W for attention processing
                _, (H, W) = actual_baseline.vit.prepare_tokens_with_masks(images)
                
                # Baseline predictions with attention
                logits1, attn1, _ = model_baseline(images, True)
                preds1 = logits1.argmax(dim=1)
                probs1 = F.softmax(logits1, dim=1)
                
                # DAGA predictions with attention
                logits2, attn2, _ = model_daga(images, True)
                preds2 = logits2.argmax(dim=1)
                probs2 = F.softmax(logits2, dim=1)
                
                # Find cases where baseline wrong, DAGA correct
                mask = (preds1 != labels) & (preds2 == labels)
                
                if mask.sum() > 0:
                    for idx in torch.where(mask)[0]:
                        gt_prob1 = probs1[idx, labels[idx]].item()
                        gt_prob2 = probs2[idx, labels[idx]].item()
                        
                        # Process attention maps with correct H, W
                        attn_map1 = self._process_attention(attn1, idx, H, W) if attn1 is not None else None
                        attn_map2 = self._process_attention(attn2, idx, H, W) if attn2 is not None else None
                        
                        improvements.append({
                            'image': images[idx].cpu(),
                            'label': labels[idx].item(),
                            'pred1': preds1[idx].item(),
                            'pred2': preds2[idx].item(),
                            'prob_diff': gt_prob2 - gt_prob1,
                            'prob1': gt_prob1,
                            'prob2': gt_prob2,
                            'attn1': attn_map1,
                            'attn2': attn_map2
                        })
        
        if not improvements:
            print("⚠️ No improvements found")
            return
            
        # Sort by probability difference
        improvements.sort(key=lambda x: x['prob_diff'], reverse=True)
        top_improvements = improvements[:self.args.num_visualize]
        
        print(f"\n✓ Found {len(improvements)} improvements, visualizing top {len(top_improvements)}")
        
        # Visualize
        self._visualize_classification(top_improvements, val_dataset.dataset.classes if hasattr(val_dataset, 'dataset') else None)
        
    def _visualize_classification(self, improvements, class_names):
        """Visualize classification improvements with attention maps"""
        output_dir = self.output_dir / "classification"
        output_dir.mkdir(exist_ok=True)
        
        for i, item in enumerate(improvements):
            # Create 2x2 grid: [Image, Image+Attn] x [Baseline, DAGA]
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Denormalize image
            img = item['image'].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            
            # Get labels
            true_label = class_names[item['label']] if class_names else str(item['label'])
            pred1_label = class_names[item['pred1']] if class_names else str(item['pred1'])
            pred2_label = class_names[item['pred2']] if class_names else str(item['pred2'])
            
            fig.suptitle(
                f"Classification Improvement #{i+1}\n"
                f"True Label: {true_label} | Confidence Gain: {item['prob_diff']:.3f}",
                fontsize=16, fontweight='bold'
            )
            
            # Row 1: Baseline
            # Original image
            axes[0, 0].imshow(img)
            axes[0, 0].set_title(
                f"Baseline ✗\nPred: {pred1_label}\nConf: {item['prob1']:.3f}",
                color='red', fontsize=12, fontweight='bold'
            )
            axes[0, 0].axis('off')
            
            # Baseline attention
            if item['attn1'] is not None:
                axes[0, 1].imshow(img)
                attn_resized = self._resize_attention(item['attn1'], img.shape[:2])
                im1 = axes[0, 1].imshow(attn_resized, cmap='jet', alpha=0.5)
                axes[0, 1].set_title("Baseline Attention", fontsize=12)
                axes[0, 1].axis('off')
                
                # Pure attention map
                im1_pure = axes[0, 2].imshow(item['attn1'], cmap='jet')
                axes[0, 2].set_title("Baseline Attention Map", fontsize=12)
                axes[0, 2].axis('off')
                plt.colorbar(im1_pure, ax=axes[0, 2], fraction=0.046, pad=0.04)
            else:
                axes[0, 1].text(0.5, 0.5, 'No Attention', ha='center', va='center')
                axes[0, 1].axis('off')
                axes[0, 2].axis('off')
            
            # Row 2: DAGA
            # Original image
            axes[1, 0].imshow(img)
            axes[1, 0].set_title(
                f"DAGA ✓\nPred: {pred2_label}\nConf: {item['prob2']:.3f}",
                color='green', fontsize=12, fontweight='bold'
            )
            axes[1, 0].axis('off')
            
            # DAGA attention
            if item['attn2'] is not None:
                axes[1, 1].imshow(img)
                attn_resized = self._resize_attention(item['attn2'], img.shape[:2])
                im2 = axes[1, 1].imshow(attn_resized, cmap='jet', alpha=0.5)
                axes[1, 1].set_title("DAGA Attention (Enhanced)", fontsize=12)
                axes[1, 1].axis('off')
                
                # Pure attention map
                im2_pure = axes[1, 2].imshow(item['attn2'], cmap='jet')
                axes[1, 2].set_title("DAGA Attention Map", fontsize=12)
                axes[1, 2].axis('off')
                plt.colorbar(im2_pure, ax=axes[1, 2], fraction=0.046, pad=0.04)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Attention', ha='center', va='center')
                axes[1, 1].axis('off')
                axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"improvement_{i+1}_diff_{item['prob_diff']:.3f}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Saved to: {output_dir}")
        
    def _resize_attention(self, attn_map, target_size):
        """Resize attention map to match image size"""
        from scipy.ndimage import zoom
        h_ratio = target_size[0] / attn_map.shape[0]
        w_ratio = target_size[1] / attn_map.shape[1]
        return zoom(attn_map, (h_ratio, w_ratio), order=1)
        
    def compare_segmentation(self):
        """Compare segmentation models"""
        print("\n" + "="*70)
        print("🎨 Segmentation Comparison")
        print("="*70)
        
        # Load dataset
        _, val_dataset, num_classes = get_segmentation_dataset(self.args)
        
        # Limit dataset
        if len(val_dataset) > self.args.max_samples:
            indices = list(range(self.args.max_samples))
            val_dataset = Subset(val_dataset, indices)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Load models
        model_baseline = self.load_segmentation_model(
            self.args.baseline_path, num_classes
        )
        model_daga = self.load_segmentation_model(
            self.args.daga_path, num_classes
        )
        
        # Get actual models for H, W extraction
        from torch.nn.parallel import DataParallel
        actual_baseline = model_baseline.module if isinstance(model_baseline, DataParallel) else model_baseline
        actual_daga = model_daga.module if isinstance(model_daga, DataParallel) else model_daga
        
        # Find improvements
        improvements = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Comparing"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Get H, W for attention processing
                _, (H, W) = actual_baseline.vit.prepare_tokens_with_masks(images)
                
                # Baseline predictions with attention
                logits1, attn1, _ = model_baseline(images, True)
                preds1 = logits1.argmax(dim=1)
                
                # DAGA predictions with attention
                logits2, attn2, _ = model_daga(images, True)
                preds2 = logits2.argmax(dim=1)
                
                # Calculate mIoU for each sample
                for idx in range(images.size(0)):
                    miou1 = calculate_miou(preds1[idx], masks[idx], num_classes)
                    miou2 = calculate_miou(preds2[idx], masks[idx], num_classes)
                    
                    if miou2 > miou1:  # DAGA better
                        # Process attention maps with correct H, W
                        attn_map1 = self._process_attention(attn1, idx, H, W) if attn1 is not None else None
                        attn_map2 = self._process_attention(attn2, idx, H, W) if attn2 is not None else None
                        
                        improvements.append({
                            'image': images[idx].cpu(),
                            'mask_gt': masks[idx].cpu(),
                            'mask_pred1': preds1[idx].cpu(),
                            'mask_pred2': preds2[idx].cpu(),
                            'miou_diff': (miou2 - miou1) * 100,
                            'miou1': miou1 * 100,
                            'miou2': miou2 * 100,
                            'attn1': attn_map1,
                            'attn2': attn_map2
                        })
        
        if not improvements:
            print("⚠️ No improvements found")
            return
            
        # Sort by mIoU difference
        improvements.sort(key=lambda x: x['miou_diff'], reverse=True)
        top_improvements = improvements[:self.args.num_visualize]
        
        print(f"\n✓ Found {len(improvements)} improvements, visualizing top {len(top_improvements)}")
        
        # Visualize
        self._visualize_segmentation(top_improvements)
        
    def _visualize_segmentation(self, improvements):
        """Visualize segmentation improvements with attention maps"""
        output_dir = self.output_dir / "segmentation"
        output_dir.mkdir(exist_ok=True)
        
        for i, item in enumerate(improvements):
            # Create two separate figures: one for segmentation results, one for attention maps
            
            # ===== Figure 1: Segmentation Results (2x2 grid) =====
            fig1, axes1 = plt.subplots(2, 2, figsize=(14, 14))
            
            # Denormalize image
            img = item['image'].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            
            fig1.suptitle(
                f"Segmentation Results - Improvement #{i+1}\n"
                f"mIoU Gain: {item['miou_diff']:.2f}% ({item['miou1']:.2f}% → {item['miou2']:.2f}%)",
                fontsize=16, fontweight='bold'
            )
            
            # Row 1: Original image and Ground Truth
            axes1[0, 0].imshow(img)
            axes1[0, 0].set_title("Original Image", fontsize=13, fontweight='bold')
            axes1[0, 0].axis('off')
            
            gt_mask = item['mask_gt'].numpy()
            gt_mask_display = np.ma.masked_where(gt_mask == 255, gt_mask)
            axes1[0, 1].imshow(gt_mask_display, cmap='nipy_spectral', vmin=0, vmax=149)
            axes1[0, 1].set_title("Ground Truth", fontsize=13, fontweight='bold')
            axes1[0, 1].axis('off')
            
            # Row 2: Baseline and DAGA predictions
            pred1_mask = item['mask_pred1'].numpy()
            axes1[1, 0].imshow(pred1_mask, cmap='nipy_spectral', vmin=0, vmax=149)
            axes1[1, 0].set_title(f"Baseline Prediction ✗\nmIoU: {item['miou1']:.2f}%", 
                                color='red', fontsize=13, fontweight='bold')
            axes1[1, 0].axis('off')
            
            pred2_mask = item['mask_pred2'].numpy()
            axes1[1, 1].imshow(pred2_mask, cmap='nipy_spectral', vmin=0, vmax=149)
            axes1[1, 1].set_title(f"DAGA Prediction ✓\nmIoU: {item['miou2']:.2f}%",
                                color='green', fontsize=13, fontweight='bold')
            axes1[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"improvement_{i+1}_miou_gain_{item['miou_diff']:.2f}_segmentation.png",
                       dpi=150, bbox_inches='tight')
            plt.close(fig1)
            
            # ===== Figure 2: Attention Maps (2x3 grid) =====
            fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
            
            fig2.suptitle(
                f"Attention Maps - Improvement #{i+1}\n"
                f"mIoU Gain: {item['miou_diff']:.2f}%",
                fontsize=16, fontweight='bold'
            )
            
            # Row 1: Baseline attention
            axes2[0, 0].imshow(img)
            axes2[0, 0].set_title(f"Baseline\nmIoU: {item['miou1']:.2f}%", 
                                 color='red', fontsize=12, fontweight='bold')
            axes2[0, 0].axis('off')
            
            if item['attn1'] is not None:
                # Baseline attention overlay
                axes2[0, 1].imshow(img)
                attn_resized = self._resize_attention(item['attn1'], img.shape[:2])
                im1 = axes2[0, 1].imshow(attn_resized, cmap='jet', alpha=0.5)
                axes2[0, 1].set_title("Baseline Attention Overlay", fontsize=12)
                axes2[0, 1].axis('off')
                
                # Pure baseline attention map
                im1_pure = axes2[0, 2].imshow(item['attn1'], cmap='jet')
                axes2[0, 2].set_title("Baseline Attention Map", fontsize=12)
                axes2[0, 2].axis('off')
                plt.colorbar(im1_pure, ax=axes2[0, 2], fraction=0.046, pad=0.04)
            else:
                axes2[0, 1].text(0.5, 0.5, 'No Attention', ha='center', va='center')
                axes2[0, 1].axis('off')
                axes2[0, 2].axis('off')
            
            # Row 2: DAGA attention
            axes2[1, 0].imshow(img)
            axes2[1, 0].set_title(f"DAGA\nmIoU: {item['miou2']:.2f}%",
                                 color='green', fontsize=12, fontweight='bold')
            axes2[1, 0].axis('off')
            
            if item['attn2'] is not None:
                # DAGA attention overlay
                axes2[1, 1].imshow(img)
                attn_resized = self._resize_attention(item['attn2'], img.shape[:2])
                im2 = axes2[1, 1].imshow(attn_resized, cmap='jet', alpha=0.5)
                axes2[1, 1].set_title("DAGA Attention Overlay (Enhanced)", fontsize=12)
                axes2[1, 1].axis('off')
                
                # Pure DAGA attention map
                im2_pure = axes2[1, 2].imshow(item['attn2'], cmap='jet')
                axes2[1, 2].set_title("DAGA Attention Map", fontsize=12)
                axes2[1, 2].axis('off')
                plt.colorbar(im2_pure, ax=axes2[1, 2], fraction=0.046, pad=0.04)
            else:
                axes2[1, 1].text(0.5, 0.5, 'No Attention', ha='center', va='center')
                axes2[1, 1].axis('off')
                axes2[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"improvement_{i+1}_miou_gain_{item['miou_diff']:.2f}_attention.png",
                       dpi=150, bbox_inches='tight')
            plt.close(fig2)
        
        print(f"✓ Saved to: {output_dir}")
        
    def run(self):
        """Run comparison based on task"""
        if self.args.task == "classification":
            self.compare_classification()
        elif self.args.task == "segmentation":
            self.compare_segmentation()
        elif self.args.task == "detection":
            print("⚠️ Detection comparison not yet implemented")
        else:
            raise ValueError(f"Unknown task: {self.args.task}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Baseline and DAGA Models")
    
    # Task and paths
    parser.add_argument("--task", choices=["classification", "detection", "segmentation"],
                       required=True, help="Task to compare")
    parser.add_argument("--baseline_path", type=str, required=True,
                       help="Path to baseline model checkpoint")
    parser.add_argument("--daga_path", type=str, required=True,
                       help="Path to DAGA model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./visualization/results",
                       help="Output directory for visualizations")
    
    # Model config
    parser.add_argument("--model_name", type=str, default="dinov3_vits16")
    parser.add_argument("--pretrained_path", type=str,
                       default="/home/user/zhoutianjian/DAGA/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    
    # Dataset config
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--input_size", type=int, default=518)
    
    # Comparison config
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=500,
                       help="Max samples to evaluate")
    parser.add_argument("--num_visualize", type=int, default=10,
                       help="Number of top improvements to visualize")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "="*70)
    print("🔍 Model Comparison Tool")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Baseline: {args.baseline_path}")
    print(f"DAGA: {args.daga_path}")
    print(f"Output: {args.output_dir}")
    print("="*70)
    
    comparer = ModelComparer(args)
    comparer.run()
    
    print("\n✅ Comparison complete!")


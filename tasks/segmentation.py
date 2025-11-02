import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import swanlab
from PIL import Image

from core.daga import DAGA
from core.heads import LinearSegmentationHead
from core.backbones import get_attention_map, compute_daga_guidance_map, process_attention_weights
from core.utils import get_base_model


class SegmentationModel(nn.Module):
    def __init__(
        self,
        pretrained_vit,
        num_classes=150,
        use_daga=False,
        daga_layers=[11],
        out_indices=[2, 5, 8, 11],
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.num_classes = num_classes
        self.use_daga = use_daga
        self.daga_layers = daga_layers
        self.out_indices = out_indices
        self.feature_dim = self.vit.embed_dim
        self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
        
        self.num_storage_tokens = -1
        self.captured_guidance_attn = None
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        if self.use_daga:
            self.daga_modules = nn.ModuleDict(
                {str(i): DAGA(feature_dim=self.feature_dim) for i in daga_layers}
            )
            for param in self.daga_modules.parameters():
                param.requires_grad = True
        
        embed_dims = [self.feature_dim] * len(out_indices)
        self.decode_head = LinearSegmentationHead(
            in_channels=embed_dims,
            num_classes=num_classes
        )
        
        for param in self.decode_head.parameters():
            param.requires_grad = True
        
        print(
            f"✓ SegmentationModel initialized:\n"
            f"  - Feature dim: {self.feature_dim}\n"
            f"  - Num classes: {num_classes}\n"
            f"  - Out indices: {out_indices}\n"
            f"  - Use DAGA: {self.use_daga} (Layers: {self.daga_layers if self.use_daga else 'N/A'})"
        )
    
    def forward(self, x, request_visualization_maps=False):
        B = x.shape[0]
        input_size = (x.shape[2], x.shape[3])
        
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        if self.num_storage_tokens == -1:
            self.num_storage_tokens = num_registers
        
        daga_guidance_map = None
        adapted_attn_weights = None
        
        if self.use_daga:
            daga_guidance_map = compute_daga_guidance_map(
                self.vit, x_processed, H, W, self.daga_guidance_layer_idx
            )
        
        intermediate_features = []
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            if request_visualization_maps and idx == self.daga_guidance_layer_idx:
                with torch.no_grad():
                    adapted_attn_weights = get_attention_map(block, x_processed)
            
            x_processed = block(x_processed, rope_sincos)
            
            if (
                self.use_daga
                and idx in self.daga_layers
                and daga_guidance_map is not None
            ):
                cls_token = x_processed[:, :1, :]
                register_start_index = 1
                register_end_index = 1 + num_registers
                register_tokens = x_processed[:, register_start_index:register_end_index, :]
                patch_start_index = 1 + num_registers
                patch_tokens = x_processed[:, patch_start_index:, :]
                
                adapted_patch_tokens = self.daga_modules[str(idx)](
                    patch_tokens, daga_guidance_map
                )
                
                x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
            
            if idx in self.out_indices:
                patch_features = x_processed[:, 1 + num_registers:, :]
                feat_spatial = patch_features.transpose(1, 2).reshape(B, C, H, W)
                intermediate_features.append(feat_spatial)
        
        seg_logits = self.decode_head(intermediate_features, input_size)
        
        return seg_logits, adapted_attn_weights, daga_guidance_map


def setup_training_components(model, args):
    """Setup criterion, optimizer, scheduler for segmentation"""
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    base_model = model.module if isinstance(model, DataParallel) else model
    
    daga_params = []
    head_params = []
    
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if "daga" in name:
                daga_params.append(param)
            else:
                head_params.append(param)
    
    lr_scaled = args.lr * (args.batch_size * torch.cuda.device_count()) / 256.0
    
    param_groups = [{"params": head_params, "lr": lr_scaled, "weight_decay": args.weight_decay}]
    if daga_params:
        param_groups.append(
            {"params": daga_params, "lr": lr_scaled * 0.5, "weight_decay": args.weight_decay}
        )
    
    optimizer = torch.optim.AdamW(param_groups)
    
    warmup_epochs = 1
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Avoid division by zero for single epoch training
        if args.epochs <= warmup_epochs:
            return 1.0
        return 0.5 * (
            1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return criterion, optimizer, scheduler


def calculate_miou(pred, target, num_classes, ignore_index=255):
    """Calculate mean IoU"""
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        valid_mask = (target != ignore_index)
        pred_cls = pred_cls & valid_mask
        target_cls = target_cls & valid_mask
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union == 0:
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    num_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        # Call with positional argument to avoid DataParallel kwargs issue
        logits, _, _ = model(images, False)
        
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            for i in range(images.size(0)):
                miou = calculate_miou(preds[i], masks[i], model.module.num_classes if isinstance(model, DataParallel) else model.num_classes)
                total_miou += miou
                num_samples += 1
        
        running_avg_loss = total_loss / (batch_idx + 1)
        running_avg_miou = (total_miou / num_samples * 100) if num_samples > 0 else 0
        
        pbar.set_postfix(
            {
                "Loss": f"{running_avg_loss:.4f}",
                "mIoU": f"{running_avg_miou:.2f}%",
            }
        )
    
    return total_loss / len(dataloader), (total_miou / num_samples * 100) if num_samples > 0 else 0


def evaluate(model, dataloader, device, num_classes):
    """Evaluate segmentation model"""
    model.eval()
    
    total_miou = 0.0
    total_pixel_acc = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Call with positional argument to avoid DataParallel kwargs issue
            logits, _, _ = model(images, False)
            preds = logits.argmax(dim=1)
            
            for i in range(images.size(0)):
                miou = calculate_miou(preds[i], masks[i], num_classes)
                total_miou += miou
                
                valid_mask = (masks[i] != 255)
                if valid_mask.sum() > 0:
                    pixel_acc = (preds[i][valid_mask] == masks[i][valid_mask]).float().mean()
                    total_pixel_acc += pixel_acc.item()
                
                num_samples += 1
    
    mean_miou = (total_miou / num_samples * 100) if num_samples > 0 else 0.0
    mean_pixel_acc = (total_pixel_acc / num_samples * 100) if num_samples > 0 else 0.0
    
    return mean_miou, mean_pixel_acc


def visualize_segmentation_results(
    model, fixed_images, fixed_masks, args, output_dir, epoch, colormap=None
):
    """Visualize segmentation predictions with attention maps (separated into two groups)"""
    if fixed_images is None:
        return []
    
    base_model = get_base_model(model)
    base_model.eval()
    vis_figs = []
    
    with torch.no_grad():
        _, (H, W) = base_model.vit.prepare_tokens_with_masks(fixed_images)
        num_patches_expected = H * W
        
        logits, adapted_attn_weights, _ = base_model(
            fixed_images, True
        )
        
        preds = logits.argmax(dim=1)
        
        adapted_attn_np = None
        baseline_attn_np = None
        
        if adapted_attn_weights is not None:
            adapted_attn_np = process_attention_weights(adapted_attn_weights, num_patches_expected, H, W)
            
            x_proc, _ = base_model.vit.prepare_tokens_with_masks(fixed_images)
            baseline_raw_weights = None
            for i in range(base_model.daga_guidance_layer_idx + 1):
                rope_sincos = (
                    base_model.vit.rope_embed(H=H, W=W)
                    if base_model.vit.rope_embed
                    else None
                )
                if i == base_model.daga_guidance_layer_idx:
                    baseline_raw_weights = get_attention_map(base_model.vit.blocks[i], x_proc)
                x_proc = base_model.vit.blocks[i](x_proc, rope_sincos)
            
            if baseline_raw_weights is not None:
                baseline_attn_np = process_attention_weights(baseline_raw_weights, num_patches_expected, H, W)
        
        images_np = fixed_images.cpu().numpy()
        masks_np = fixed_masks.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        vis_save_path = Path(output_dir) / "visualizations"
        vis_save_path.mkdir(parents=True, exist_ok=True)
        
        for j in range(images_np.shape[0]):
            # Group 1: Segmentation Results (Image, GT, Prediction)
            fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
            fig1.suptitle(f"Segmentation Results - Epoch {epoch+1} - Sample {j}", fontsize=14, fontweight="bold")
            
            img = images_np[j].transpose(1, 2, 0)
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            axes1[0].imshow(img)
            axes1[0].set_title("Original Image")
            axes1[0].axis("off")
            
            axes1[1].imshow(masks_np[j], cmap='tab20')
            axes1[1].set_title("Ground Truth")
            axes1[1].axis("off")
            
            axes1[2].imshow(preds_np[j], cmap='tab20')
            axes1[2].set_title("Prediction")
            axes1[2].axis("off")
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            vis_figs.append(fig1)
            
            fig1.savefig(
                vis_save_path / f"epoch_{epoch+1}_sample_{j}_segmentation.png",
                dpi=100,
            )
            plt.close(fig1)
            
            # Group 2: Attention Map Comparison (only if DAGA is used)
            if adapted_attn_np is not None and baseline_attn_np is not None:
                fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
                fig2.suptitle(f"Attention Maps - Epoch {epoch+1} - Sample {j}", fontsize=14, fontweight="bold")
                
                axes2[0].imshow(baseline_attn_np[j], cmap="viridis")
                axes2[0].set_title("Frozen Backbone Attn")
                axes2[0].axis("off")
                
                axes2[1].imshow(adapted_attn_np[j], cmap="viridis")
                axes2[1].set_title("Adapted Model Attn")
                axes2[1].axis("off")
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                vis_figs.append(fig2)
                
                fig2.savefig(
                    vis_save_path / f"epoch_{epoch+1}_sample_{j}_attention.png",
                    dpi=100,
                )
                plt.close(fig2)
    
    return vis_figs


def prepare_visualization_data(val_dataset, args, device):
    """Prepare fixed batch for visualization"""
    if not args.enable_visualization:
        return None, None
    
    print(f"\n📸 Preparing visualization data...")
    indices = list(range(min(args.num_vis_samples, len(val_dataset))))
    vis_subset = torch.utils.data.Subset(val_dataset, indices)
    vis_loader = DataLoader(vis_subset, batch_size=len(indices), shuffle=False)
    fixed_images, fixed_masks = next(iter(vis_loader))
    print("✓ Visualization data loaded.")
    return fixed_images.to(device), fixed_masks.to(device)


def run_training_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    args,
    output_dir,
    fixed_vis_images,
    fixed_vis_masks,
    num_classes,
):
    """Execute main training and evaluation loop"""
    best_miou = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss, train_miou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_miou, val_pixel_acc = evaluate(model, val_loader, device, num_classes)
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        print(f"\n📈 Epoch {epoch+1}/{args.epochs} Summary:")
        print(
            f"   Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.2f}%"
        )
        print(
            f"   Val mIoU: {val_miou:.2f}% | Val Pixel Acc: {val_pixel_acc:.2f}%"
        )
        print(f"   Time Elapsed: {elapsed_time/60:.1f}min")
        
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "val_miou": val_miou,
            "val_pixel_acc": val_pixel_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "total_time_minutes": elapsed_time / 60,
        }
        
        if args.enable_visualization and (
            epoch % args.log_freq == 0 or epoch == args.epochs - 1
        ):
            print("📊 Generating segmentation visualizations...")
            vis_figs = visualize_segmentation_results(
                model, fixed_vis_images, fixed_vis_masks, args, output_dir, epoch
            )
            if vis_figs:
                log_dict["segmentation_results"] = [
                    swanlab.Image(fig) for fig in vis_figs
                ]
        
        swanlab.log(log_dict, step=epoch + 1) if getattr(args, 'enable_swanlab', True) else None
        
        if val_miou > best_miou:
            best_miou = val_miou
            save_path = output_dir / "best_model.pth"
            from core.utils import save_checkpoint
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_miou,
                args,
                save_path,
            )
            print(f"   ✅ New best model saved! (Val mIoU: {best_miou:.2f}%)")
    
    return best_miou, val_miou, (time.time() - start_time) / 60

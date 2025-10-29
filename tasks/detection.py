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
import matplotlib.patches as patches
import swanlab

from core.daga import DAGA
from core.heads import DetectionHead
from core.backbones import get_attention_map, compute_daga_guidance_map, process_attention_weights
from core.utils import get_base_model


class DetectionModel(nn.Module):
    def __init__(
        self,
        pretrained_vit,
        num_classes=80,
        use_daga=False,
        daga_layers=[11],
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.num_classes = num_classes
        self.use_daga = use_daga
        self.daga_layers = daga_layers
        self.feature_dim = self.vit.embed_dim
        self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
        
        self.num_storage_tokens = -1
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        if self.use_daga:
            self.daga_modules = nn.ModuleDict(
                {str(i): DAGA(feature_dim=self.feature_dim) for i in daga_layers}
            )
            for param in self.daga_modules.parameters():
                param.requires_grad = True
        
        self.detection_head = DetectionHead(self.feature_dim, num_classes)
        
        for param in self.detection_head.parameters():
            param.requires_grad = True
        
        print(
            f"✓ DetectionModel initialized:\n"
            f"  - Feature dim: {self.feature_dim}\n"
            f"  - Num classes: {num_classes}\n"
            f"  - Use DAGA: {self.use_daga} (Layers: {self.daga_layers if self.use_daga else 'N/A'})"
        )
    
    def forward(self, x, request_visualization_maps=False):
        B = x.shape[0]
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
        
        patch_features = x_processed[:, 1 + num_registers:, :]
        feat_spatial = patch_features.transpose(1, 2).reshape(B, C, H, W)
        
        cls_output, bbox_output, obj_output = self.detection_head(feat_spatial)
        
        return cls_output, bbox_output, obj_output, adapted_attn_weights, daga_guidance_map


def setup_training_components(model, args):
    """Setup optimizer, scheduler for detection"""
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
        return 0.5 * (
            1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def detection_loss(cls_pred, bbox_pred, obj_pred, boxes_list, labels_list):
    """Simplified detection loss (placeholder for real YOLO/FCOS loss)"""
    batch_size = len(boxes_list)
    
    total_loss = 0.0
    for b in range(batch_size):
        if len(boxes_list[b]) > 0:
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_pred[b].mean(), 
                torch.ones(1, device=obj_pred.device) * 0.5
            )
            total_loss += obj_loss
        else:
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_pred[b].mean(), 
                torch.zeros(1, device=obj_pred.device)
            )
            total_loss += obj_loss
    
    return total_loss / batch_size


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, (images, boxes_list, labels_list) in enumerate(pbar):
        images = images.to(device)
        optimizer.zero_grad()
        
        cls_pred, bbox_pred, obj_pred, _, _ = model(images, request_visualization_maps=False)
        
        loss = detection_loss(cls_pred, bbox_pred, obj_pred, boxes_list, labels_list)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        running_avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({"Loss": f"{running_avg_loss:.4f}"})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate detection model (simplified)"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, boxes_list, labels_list in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            cls_pred, bbox_pred, obj_pred, _, _ = model(images, request_visualization_maps=False)
            
            loss = detection_loss(cls_pred, bbox_pred, obj_pred, boxes_list, labels_list)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def visualize_detection_results(
    model, fixed_images, fixed_boxes_list, args, output_dir, epoch
):
    """Visualize detection predictions with attention maps"""
    if fixed_images is None:
        return []
    
    base_model = get_base_model(model)
    base_model.eval()
    vis_figs = []
    
    with torch.no_grad():
        _, (H, W) = base_model.vit.prepare_tokens_with_masks(fixed_images)
        num_patches_expected = H * W
        
        cls_pred, bbox_pred, obj_pred, adapted_attn_weights, _ = base_model(
            fixed_images, request_visualization_maps=True
        )
        
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
        
        vis_save_path = Path(output_dir) / "visualizations"
        vis_save_path.mkdir(parents=True, exist_ok=True)
        
        for j in range(images_np.shape[0]):
            ncols = 3 if (adapted_attn_np is not None and baseline_attn_np is not None) else 1
            fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))
            if ncols == 1:
                axes = [axes]
            fig.suptitle(f"Epoch {epoch+1} - Sample {j}", fontsize=14, fontweight="bold")
            
            img = images_np[j].transpose(1, 2, 0)
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            axes[0].imshow(img)
            
            if j < len(fixed_boxes_list):
                boxes = fixed_boxes_list[j]
                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle(
                            (x1*img.shape[1], y1*img.shape[0]), 
                            (x2-x1)*img.shape[1], 
                            (y2-y1)*img.shape[0],
                            linewidth=2, edgecolor='r', facecolor='none'
                        )
                        axes[0].add_patch(rect)
            
            axes[0].set_title("Image with GT Boxes")
            axes[0].axis("off")
            
            if adapted_attn_np is not None and baseline_attn_np is not None:
                axes[1].imshow(baseline_attn_np[j], cmap="viridis")
                axes[1].set_title("Frozen Backbone Attn")
                axes[1].axis("off")
                
                axes[2].imshow(adapted_attn_np[j], cmap="viridis")
                axes[2].set_title("Adapted Model Attn")
                axes[2].axis("off")
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            vis_figs.append(fig)
            
            fig.savefig(
                vis_save_path / f"epoch_{epoch+1}_sample_{j}.png",
                dpi=100,
            )
            plt.close(fig)
    
    return vis_figs


def prepare_visualization_data(val_dataset, args, device):
    """Prepare fixed batch for visualization"""
    if not args.enable_visualization:
        return None, None
    
    print(f"\n📸 Preparing visualization data...")
    indices = list(range(min(args.num_vis_samples, len(val_dataset))))
    vis_subset = torch.utils.data.Subset(val_dataset, indices)
    vis_loader = DataLoader(vis_subset, batch_size=len(indices), shuffle=False, 
                           collate_fn=lambda x: (
                               torch.stack([item[0] for item in x]),
                               [item[1] for item in x],
                               [item[2] for item in x]
                           ))
    fixed_images, fixed_boxes, fixed_labels = next(iter(vis_loader))
    print("✓ Visualization data loaded.")
    return fixed_images.to(device), fixed_boxes


def run_training_loop(
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
):
    """Execute main training and evaluation loop"""
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        print(f"\n📈 Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Time Elapsed: {elapsed_time/60:.1f}min")
        
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "total_time_minutes": elapsed_time / 60,
        }
        
        if args.enable_visualization and (
            epoch % args.log_freq == 0 or epoch == args.epochs - 1
        ):
            print("📊 Generating detection visualizations...")
            vis_figs = visualize_detection_results(
                model, fixed_vis_images, fixed_vis_boxes, args, output_dir, epoch
            )
            if vis_figs:
                log_dict["detection_results"] = [
                    swanlab.Image(fig) for fig in vis_figs
                ]
        
        swanlab.log(log_dict, step=epoch + 1)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = output_dir / "best_model.pth"
            from core.utils import save_checkpoint
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_loss,
                args,
                save_path,
            )
            print(f"   ✅ New best model saved! (Val Loss: {best_loss:.4f})")
    
    return best_loss, val_loss, (time.time() - start_time) / 60

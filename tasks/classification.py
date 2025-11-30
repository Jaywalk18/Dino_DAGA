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

from core.daga import DAGA
from core.backbones import get_attention_map, compute_daga_guidance_map, process_attention_weights
from core.utils import get_base_model


class ClassificationModel(nn.Module):
    def __init__(
        self,
        pretrained_vit,
        num_classes=10,
        use_daga=False,
        daga_layers=[11],
        enable_visualization=False,
        vis_attn_layer=11,
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
        
        self.num_storage_tokens = -1
        self.captured_attn = None
        self.captured_guidance_attn = None
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        if self.use_daga:
            self.daga_modules = nn.ModuleDict(
                {str(i): DAGA(feature_dim=self.feature_dim) for i in daga_layers}
            )
        
        # Use simple linear layer like raw_code (not ClassificationHead)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        # Ensure classifier parameters are trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        if self.use_daga:
            for param in self.daga_modules.parameters():
                param.requires_grad = True
        
        print(
            f"✓ ClassificationModel initialized:\n"
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
        if self.use_daga:
            daga_guidance_map = compute_daga_guidance_map(
                self.vit, x_processed, H, W, self.daga_guidance_layer_idx
            )
        
        adapted_attn_weights = None
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # Extract attention BEFORE block forward (like raw_code)
            # This captures how the current features (potentially DAGA-adapted from previous layers) 
            # affect this block's attention
            if request_visualization_maps and idx == self.vis_attn_layer:
                with torch.no_grad():
                    adapted_attn_weights = get_attention_map(block, x_processed)
            
            # Pass tokens through the block
            x_processed = block(x_processed, rope_sincos)
            
            # Apply DAGA AFTER block forward (modifies block output)
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
                
                # Store original for comparison
                patch_tokens_before = patch_tokens.clone() if request_visualization_maps and idx == self.vis_attn_layer else None
                
                adapted_patch_tokens = self.daga_modules[str(idx)](
                    patch_tokens, daga_guidance_map
                )
                
                # Debug: Print feature change magnitude during visualization
                if request_visualization_maps and idx == self.vis_attn_layer:
                    feature_diff = (adapted_patch_tokens - patch_tokens_before).abs().mean().item()
                    mix_weight = self.daga_modules[str(idx)].mix_weight.item()
                    print(f"  [Layer {idx}] DAGA feature change: {feature_diff:.6f}, mix_weight: {mix_weight:.6f}")
                
                x_processed = torch.cat([cls_token, register_tokens, adapted_patch_tokens], dim=1)
        
        x_normalized = self.vit.norm(x_processed)
        # Use only CLS token for classification (matching raw_code)
        features = x_normalized[:, 0]  # (B, C)
        logits = self.classifier(features)
        
        return logits, adapted_attn_weights, daga_guidance_map


def setup_training_components(model, args, world_size=1):
    """Setup criterion, optimizer, scheduler
    
    Args:
        model: The model to train
        args: Training arguments
        world_size: Number of GPUs in DDP training (default: 1)
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Handle both DataParallel and DistributedDataParallel
    base_model = model.module if isinstance(model, (DataParallel, DDP)) else model
    
    daga_params = []
    classifier_params = []
    
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if "daga" in name:
                daga_params.append(param)
            else:
                classifier_params.append(param)
    
    # Linear learning rate scaling rule: lr ∝ batch_size
    # Reference batch size is 256, so: lr_scaled = base_lr * (total_batch_size / 256)
    # In DDP: total_batch_size = per_gpu_batch_size * world_size
    total_batch_size = args.batch_size * world_size
    lr_scaled = args.lr * total_batch_size / 256.0
    
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
        # Avoid division by zero for single epoch training
        if args.epochs <= warmup_epochs:
            return 1.0
        return 0.5 * (
            1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return criterion, optimizer, scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
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
    """Evaluate model"""
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


def visualize_attention_comparison(
    model, fixed_images, args, output_dir, epoch, test_dataset=None
):
    """Generate attention comparison visualizations
    
    Baseline: 1x2 layout - [Original Image, Frozen Backbone Attn]
    DAGA: 1x4 layout - [Original Image, Frozen Backbone Attn, Adapted Attn, Adapted Attn Overlay]
    """
    from scipy.ndimage import zoom
    
    if fixed_images is None:
        return []
    
    base_model = get_base_model(model)
    base_model.eval()
    vis_figs = []
    
    class_names = getattr(test_dataset, "classes", None)
    
    # Get actual model from DDP wrapper if needed
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    actual_base_model = base_model.module if isinstance(base_model, (DataParallel, DDP)) else base_model
    
    is_daga_model = getattr(actual_base_model, "use_daga", False)
    
    # Print DAGA status
    if is_daga_model:
        print(f"\n[DAGA] Status at epoch {epoch+1}:")
        print(f"  Layers: {actual_base_model.daga_layers}")
        print(f"  Mix weights:")
        for layer_idx, daga_module in actual_base_model.daga_modules.items():
            print(f"    Layer {layer_idx}: {daga_module.mix_weight.item():.6f}")
    
    with torch.no_grad():
        _, (H, W) = actual_base_model.vit.prepare_tokens_with_masks(fixed_images)
        num_patches_expected = H * W
        vis_attn_layer = actual_base_model.vis_attn_layer
        
        # Always get baseline attention (frozen backbone)
        x_proc, _ = actual_base_model.vit.prepare_tokens_with_masks(fixed_images)
        baseline_raw_weights = None
        for i in range(vis_attn_layer + 1):
            rope_sincos = (
                actual_base_model.vit.rope_embed(H=H, W=W)
                if actual_base_model.vit.rope_embed
                else None
            )
            if i == vis_attn_layer:
                baseline_raw_weights = get_attention_map(
                    actual_base_model.vit.blocks[i], x_proc
                )
            x_proc = actual_base_model.vit.blocks[i](x_proc, rope_sincos)
        
        if baseline_raw_weights is None:
            print(f"⚠ Warning: Failed to extract baseline attention from layer {vis_attn_layer}.")
            return []
        
        baseline_attn_np = process_attention_weights(baseline_raw_weights, num_patches_expected, H, W)
        if baseline_attn_np is None:
            return []
        
        # Get adapted attention and predictions (only for DAGA model)
        adapted_attn_np = None
        daga_guidance_map = None
        
        logits, adapted_attn_weights, daga_guidance_map = base_model(
            fixed_images, request_visualization_maps=True
        )
        predictions = logits.argmax(dim=1).cpu().numpy()
        
        if is_daga_model and adapted_attn_weights is not None:
            adapted_attn_np = process_attention_weights(adapted_attn_weights, num_patches_expected, H, W)
        
        # Prepare images
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
            
            # Denormalize image
            img = images_np[j].transpose(1, 2, 0)
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            
            if is_daga_model and adapted_attn_np is not None:
                # DAGA: 1x4 layout
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                fig.suptitle(fig_title, fontsize=14, fontweight="bold")
                
                # Panel 1: Original Image
                axes[0].imshow(img)
                axes[0].set_title("Original Image")
                axes[0].axis("off")
                
                # Panel 2: Frozen Backbone Attention
                im1 = axes[1].imshow(baseline_attn_np[j], cmap="viridis", vmin=0, vmax=1)
                axes[1].set_title(f"Frozen Backbone Attn (L{vis_attn_layer})")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                
                # Panel 3: Adapted Attention (after DAGA)
                im2 = axes[2].imshow(adapted_attn_np[j], cmap="viridis", vmin=0, vmax=1)
                axes[2].set_title(f"Adapted Attn (After DAGA, L{vis_attn_layer})")
                axes[2].axis("off")
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                
                # Panel 4: Adapted Attention Overlay
                axes[3].imshow(img)
                attn_map = adapted_attn_np[j]
                if attn_map.shape != img.shape[:2]:
                    zoom_h = img.shape[0] / attn_map.shape[0]
                    zoom_w = img.shape[1] / attn_map.shape[1]
                    attn_resized = zoom(attn_map, (zoom_h, zoom_w), order=1)
                else:
                    attn_resized = attn_map
                im3 = axes[3].imshow(attn_resized, cmap="jet", alpha=0.5, vmin=0, vmax=1)
                axes[3].set_title("Adapted Attn Overlay")
                axes[3].axis("off")
                plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
            else:
                # Baseline: 1x2 layout
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(fig_title, fontsize=14, fontweight="bold")
                
                # Panel 1: Original Image
                axes[0].imshow(img)
                axes[0].set_title("Original Image")
                axes[0].axis("off")
                
                # Panel 2: Frozen Backbone Attention
                im1 = axes[1].imshow(baseline_attn_np[j], cmap="viridis", vmin=0, vmax=1)
                axes[1].set_title(f"Frozen Backbone Attn (L{vis_attn_layer})")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            vis_figs.append(fig)
            
            fig.savefig(
                vis_save_path / f"epoch_{epoch+1}_imgidx_{original_image_index}.png",
                dpi=100,
            )
            plt.close(fig)
    
    return vis_figs


def prepare_visualization_data(test_dataset, args, device):
    """Prepare fixed batch of images for visualization"""
    if not args.enable_visualization:
        return None
    
    # 边界检查：确保 vis_indices 不超过测试集大小
    max_idx = len(test_dataset) - 1
    valid_indices = [idx for idx in args.vis_indices if idx <= max_idx]
    
    if len(valid_indices) < len(args.vis_indices):
        print(f"⚠️  Warning: Some vis_indices exceed test set size ({len(test_dataset)})")
        print(f"   Original indices: {args.vis_indices}")
        # 如果有无效索引，用有效的随机索引替换
        if len(valid_indices) == 0:
            # 所有索引都无效，生成新的
            import random
            valid_indices = random.sample(range(len(test_dataset)), min(4, len(test_dataset)))
        args.vis_indices = valid_indices
        print(f"   Adjusted indices: {args.vis_indices}")
    
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
    test_dataset=None,
    rank=0,
    world_size=1,
):
    """Execute main training and evaluation loop"""
    is_main_process = (rank == 0)
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        # Only print on main process
        if is_main_process:
            print(f"\n📈 Epoch {epoch+1}/{args.epochs} Summary:")
            print(
                f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
            )
            print(f"   Time Elapsed: {elapsed_time/60:.1f}min")
        
        if is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total_time_minutes": elapsed_time / 60,
            }
            
            if args.enable_visualization and fixed_vis_images is not None and (
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
            
            swanlab.log(log_dict, step=epoch + 1) if getattr(args, 'enable_swanlab', True) else None
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = output_dir / "best_model.pth"
            from core.utils import save_checkpoint
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

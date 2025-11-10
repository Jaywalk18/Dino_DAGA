"""
Detection task using simple FCOS-style detection head with DAGA support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import swanlab
import sys

# Add dinov3 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dinov3"))

from core.daga import DAGA
from core.backbones import get_attention_map, compute_daga_guidance_map, process_attention_weights
from core.utils import get_base_model
from core.simple_detection_head import (
    SimpleDetectionHead, 
    simple_detection_loss,
    decode_predictions
)


class ViTWithDAGA(nn.Module):
    """Wrapper for ViT backbone that injects DAGA between blocks"""
    def __init__(self, vit_model, use_daga=False, daga_layers=None):
        super().__init__()
        self.vit = vit_model
        self.use_daga = use_daga
        self.daga_layers = daga_layers if daga_layers else []
        self.feature_dim = vit_model.embed_dim
        self.daga_guidance_layer_idx = len(vit_model.blocks) - 1
        self.num_storage_tokens = -1
        
        # DAGA modules
        if self.use_daga:
            self.daga_modules = nn.ModuleDict({
                str(i): DAGA(feature_dim=self.feature_dim) for i in self.daga_layers
            })
            for param in self.daga_modules.parameters():
                param.requires_grad = True
        
        # Freeze original ViT
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Cache for visualization and intermediate features
        self._cached_attn = None
        self._cached_guidance = None
        self._last_intermediate_features = []
        self._last_shape = None
    
    def _forward_blocks(self, x):
        """Internal forward through blocks with optional DAGA"""
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        if self.num_storage_tokens == -1:
            self.num_storage_tokens = num_registers
        
        # Compute DAGA guidance
        daga_guidance_map = None
        if self.use_daga:
            daga_guidance_map = compute_daga_guidance_map(
                self.vit, x_processed, H, W, self.daga_guidance_layer_idx
            )
            self._cached_guidance = daga_guidance_map
        
        # Store intermediate features for get_intermediate_layers
        intermediate_features = []
        
        # Find the last DAGA layer for attention visualization
        visualization_layer = max(self.daga_layers) if self.use_daga and self.daga_layers else self.daga_guidance_layer_idx
        
        # Process through blocks
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # Cache attention BEFORE block execution (for visualization)
            # This captures the attention pattern that will be computed by this block
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
    
    def get_intermediate_layers(self, x, n, reshape=True):
        """Get intermediate layer features (required by DINOBackbone)"""
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
                
                if reshape:
                    # Extract patch tokens and reshape to spatial
                    B, seq_len, C = feat.shape
                    num_patches = H * W
                    num_registers = seq_len - num_patches - 1
                    patch_tokens = feat[:, 1+num_registers:, :]  # Skip CLS + registers
                    feat_reshaped = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
                    outputs.append(feat_reshaped)
                else:
                    outputs.append(feat)
        
        return outputs
    
    def get_intermediate_layers_with_viz(self, x, n, reshape=True, request_viz=False):
        """
        Get intermediate layer features with optional attention visualization
        This method is identical to SegmentationModel's forward logic for attention extraction
        """
        x_processed, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        B, seq_len, C = x_processed.shape
        num_patches = H * W
        num_registers = seq_len - num_patches - 1
        
        if self.num_storage_tokens == -1:
            self.num_storage_tokens = num_registers
        
        # Compute DAGA guidance
        daga_guidance_map = None
        adapted_attn_weights = None
        
        if self.use_daga:
            daga_guidance_map = compute_daga_guidance_map(
                self.vit, x_processed, H, W, self.daga_guidance_layer_idx
            )
            self._cached_guidance = daga_guidance_map
        
        intermediate_features = []
        
        # Process through blocks
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # Extract attention at daga_guidance_layer_idx BEFORE block execution
            # This matches segmentation.py line 94-96
            if request_viz and idx == self.daga_guidance_layer_idx:
                with torch.no_grad():
                    adapted_attn_weights = get_attention_map(block, x_processed)
            
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
        
        # Cache the attention for visualization
        if request_viz:
            self._cached_attn = adapted_attn_weights
        
        # Extract requested layers
        n_blocks = len(self.vit.blocks)
        if isinstance(n, int):
            indices = [n - 1]
        else:
            indices = [i - 1 if i > 0 else n_blocks + i for i in n]
        
        outputs = []
        for idx in indices:
            if idx < len(intermediate_features):
                feat = intermediate_features[idx]
                
                if reshape:
                    # Extract patch tokens and reshape to spatial
                    B, seq_len, C = feat.shape
                    patch_tokens = feat[:, 1+num_registers:, :]  # Skip CLS + registers
                    feat_reshaped = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
                    outputs.append(feat_reshaped)
                else:
                    outputs.append(feat)
        
        return outputs
    
    # Proxy attributes for compatibility
    @property
    def embed_dim(self):
        return self.vit.embed_dim
    
    @property
    def patch_size(self):
        return self.vit.patch_size
    
    @property
    def blocks(self):
        return self.vit.blocks
    
    @property
    def n_blocks(self):
        return len(self.vit.blocks)
    
    def prepare_tokens_with_masks(self, x):
        return self.vit.prepare_tokens_with_masks(x)
    
    @property
    def rope_embed(self):
        return self.vit.rope_embed


class DetectionModel(nn.Module):
    """Simple detection model using FCOS-style head with optional DAGA"""
    def __init__(self, pretrained_vit, num_classes=91, use_daga=False, daga_layers=None, layers_to_use=None):
        super().__init__()
        self.use_daga = use_daga
        self.daga_layers = daga_layers if daga_layers else []
        
        # Wrap ViT with DAGA
        self.vit_wrapper = ViTWithDAGA(pretrained_vit, use_daga, daga_layers)
        
        # Configure feature extraction layers
        n_blocks = len(self.vit_wrapper.vit.blocks)
        if layers_to_use is None:
            layers_to_use = [m * n_blocks // 4 - 1 for m in range(1, 5)]
        self.layers_to_use = layers_to_use
        
        # Feature dimension
        feature_dim = self.vit_wrapper.feature_dim * len(layers_to_use)
        
        # Simple detection head
        self.detection_head = SimpleDetectionHead(feature_dim, num_classes)
        
        # Feature stride (depends on ViT patch size)
        self.stride = self.vit_wrapper.patch_size
        
        # Freeze ViT backbone
        for param in self.vit_wrapper.parameters():
            param.requires_grad = False
        
        # Make detection head trainable
        for param in self.detection_head.parameters():
            param.requires_grad = True
        
        # Make DAGA trainable
        if self.use_daga:
            for param in self.vit_wrapper.daga_modules.parameters():
                param.requires_grad = True
    
    def forward(self, x, request_visualization_maps=False):
        """
        Forward pass
        
        Args:
            x: list of (3, H, W) tensors or stacked (B, 3, H, W) tensor
            request_visualization_maps: whether to return attention/guidance maps
            
        Returns:
            cls_logits, box_preds, centerness, attn, guidance
        """
        # Handle both list and tensor inputs
        if isinstance(x, list):
            # Stack list into tensor (all images should have same size)
            x = torch.stack(x)
        
        B, _, H, W = x.shape
        
        # Get multi-layer features through ViT with DAGA
        # This will populate vit_wrapper._cached_attn if request_visualization_maps=True
        features = self.vit_wrapper.get_intermediate_layers_with_viz(
            x, self.layers_to_use, reshape=True, request_viz=request_visualization_maps
        )
        
        # Concatenate multi-scale features
        features_concat = torch.cat(features, dim=1)  # (B, C*num_layers, H', W')
        
        # Detection head
        cls_logits, box_preds, centerness = self.detection_head(features_concat)
        
        # Get visualization if requested
        attn = self.vit_wrapper._cached_attn if request_visualization_maps else None
        guidance = self.vit_wrapper._cached_guidance if self.use_daga else None
        
        return cls_logits, box_preds, centerness, attn, guidance


def setup_training_components(model, args):
    """Setup optimizer and scheduler with official settings"""
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    base_model = model.module if isinstance(model, (DataParallel, DDP)) else model
    
    daga_params = []
    detection_params = []
    
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            if "daga" in name:
                daga_params.append(param)
            else:
                detection_params.append(param)
    
    # Learning rate scaling (official DINOv3 style)
    import torch.distributed as dist
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    lr_scaled = args.lr * (args.batch_size * world_size) / 64.0
    
    param_groups = []
    if detection_params:
        param_groups.append({
            "params": detection_params,
            "lr": lr_scaled,
            "weight_decay": getattr(args, 'weight_decay', 1e-4)
        })
    if daga_params:
        param_groups.append({
            "params": daga_params,
            "lr": lr_scaled * 0.5,
            "weight_decay": getattr(args, 'weight_decay', 1e-4) * 0.5
        })
    
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
    # Cosine schedule with warmup
    warmup_epochs = max(1, args.epochs // 20)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        if args.epochs <= warmup_epochs:
            return 1.0
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler


def detection_collate_fn(batch):
    """Collate function for detection dataloader"""
    images, boxes_list, labels_list = [], [], []
    for item in batch:
        if len(item) == 3:
            img, boxes, labels = item
        else:
            img, boxes = item
            labels = torch.zeros(len(boxes), dtype=torch.long)
        images.append(img)
        boxes_list.append(boxes)
        labels_list.append(labels)
    # Return list of images (not stacked) for NestedTensor conversion
    return images, boxes_list, labels_list


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch, num_classes):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    from tqdm import tqdm
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    
    # Get stride from model (handle DDP wrapping)
    if isinstance(model, (DataParallel, DDP)):
        stride = model.module.stride
    else:
        stride = model.stride
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, (images, boxes_list, labels_list) in enumerate(pbar):
        # images is a list of tensors, move each to device
        images = [img.to(device) for img in images]
        boxes_list = [boxes.to(device) for boxes in boxes_list]
        labels_list = [labels.to(device) for labels in labels_list]
        
        # Get image size from first image
        _, H, W = images[0].shape
        
        optimizer.zero_grad()
        cls_logits, box_preds, centerness, _, _ = model(images, False)
        
        loss_dict = simple_detection_loss(
            cls_logits, box_preds, centerness,
            boxes_list, labels_list,
            image_size=(H, W),
            stride=stride
        )
        loss = loss_dict['loss']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        running_avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({
            "Loss": f"{running_avg_loss:.4f}",
            "Cls": f"{loss_dict['cls_loss'].item():.3f}",
            "Box": f"{loss_dict['box_loss'].item():.3f}",
            "Ctr": f"{loss_dict['ctr_loss'].item():.3f}"
        })
    
    return total_loss / len(dataloader)


def compute_map(all_predictions, all_gt_boxes, all_gt_labels, num_classes, iou_threshold=0.5):
    """Compute mAP at given IoU threshold"""
    from core.simple_detection_head import box_iou
    
    aps = []
    for cls_id in range(num_classes):
        # Collect all predictions for this class
        cls_preds = []
        for img_id, pred in enumerate(all_predictions):
            mask = pred['labels'] == cls_id
            if mask.sum() > 0:
                cls_preds.append({
                    'boxes': pred['boxes'][mask],
                    'scores': pred['scores'][mask],
                    'img_id': img_id  # Store original image index
                })
        
        # Collect all GT boxes for this class
        cls_gts = []
        for gt_boxes, gt_labels in zip(all_gt_boxes, all_gt_labels):
            mask = gt_labels == cls_id
            if mask.sum() > 0:
                cls_gts.append(gt_boxes[mask])
            else:
                cls_gts.append(torch.zeros((0, 4)))
        
        # Skip if no GT for this class
        total_gt = sum(len(gt) for gt in cls_gts)
        if total_gt == 0:
            continue
        
        # Collect all predictions with scores
        all_boxes = []
        all_scores = []
        all_img_ids = []
        
        for pred in cls_preds:
            if len(pred['boxes']) > 0:
                all_boxes.append(pred['boxes'])
                all_scores.append(pred['scores'])
                # Use stored img_id instead of enumerate index
                all_img_ids.extend([pred['img_id']] * len(pred['boxes']))
        
        if len(all_boxes) == 0:
            aps.append(0.0)
            continue
        
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        # Sort by scores
        sorted_indices = torch.argsort(all_scores, descending=True)
        all_boxes = all_boxes[sorted_indices]
        all_scores = all_scores[sorted_indices]
        all_img_ids = [all_img_ids[i] for i in sorted_indices.cpu().numpy()]
        
        # Compute TP/FP
        tp = torch.zeros(len(all_boxes))
        fp = torch.zeros(len(all_boxes))
        matched = [set() for _ in range(len(cls_gts))]
        
        for i, (box, img_id) in enumerate(zip(all_boxes, all_img_ids)):
            gt_boxes = cls_gts[img_id]
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            
            ious = box_iou(box.unsqueeze(0), gt_boxes)[0]
            max_iou, max_idx = ious.max(dim=-1)  # Find max IoU among all GT boxes
            
            if max_iou >= iou_threshold and max_idx.item() not in matched[img_id]:
                tp[i] = 1
                matched[img_id].add(max_idx.item())
            else:
                fp[i] = 1
        
        # Compute precision-recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            mask = recalls >= t
            if mask.sum() > 0:
                ap += precisions[mask].max().item()
        ap /= 11.0
        
        aps.append(ap)
    
    # Return mean AP over classes that have GT (standard mAP calculation)
    return sum(aps) / len(aps) if len(aps) > 0 else 0.0


def evaluate(model, dataloader, device, num_classes):
    """Evaluate detection model with mAP"""
    model.eval()
    total_loss = 0.0
    
    from tqdm import tqdm
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    
    # Get stride from model (handle DDP wrapping)
    if isinstance(model, (DataParallel, DDP)):
        stride = model.module.stride
    else:
        stride = model.stride
    
    all_predictions = []
    all_gt_boxes = []
    all_gt_labels = []
    
    with torch.no_grad():
        for images, boxes_list, labels_list in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            boxes_list = [boxes.to(device) for boxes in boxes_list]
            labels_list = [labels.to(device) for labels in labels_list]
            
            _, H, W = images[0].shape
            
            cls_logits, box_preds, centerness, _, _ = model(images, False)
            loss_dict = simple_detection_loss(
                cls_logits, box_preds, centerness,
                boxes_list, labels_list,
                image_size=(H, W),
                stride=stride
            )
            total_loss += loss_dict['loss'].item()
            
            # Decode predictions for mAP
            detections = decode_predictions(
                cls_logits, box_preds, centerness,
                image_size=(H, W),
                stride=stride,
                score_threshold=0.05,
                nms_threshold=0.5,
                max_detections=100
            )
            
            all_predictions.extend(detections)
            all_gt_boxes.extend(boxes_list)
            all_gt_labels.extend(labels_list)
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute mAP
    mAP = compute_map(all_predictions, all_gt_boxes, all_gt_labels, num_classes, iou_threshold=0.5)
    
    return {
        'loss': avg_loss, 
        'mAP': mAP, 
        'mAP@50': mAP,
        'precision': 0.0, 
        'recall': 0.0, 
        'f1': 0.0, 
        'tp': 0, 
        'fp': 0, 
        'total_gt': 0
    }


def visualize_detection_results(model, fixed_images, fixed_boxes_list, args, output_dir, epoch, val_metrics=None):
    """Visualize detection predictions with attention maps (separated into two groups)"""
    if fixed_images is None:
        return []
    
    from core.utils import get_base_model
    from core.backbones import get_attention_map, process_attention_weights
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    
    base_model = get_base_model(model)
    base_model.eval()
    actual_base_model = base_model.module if isinstance(base_model, (DataParallel, DDP)) else base_model
    
    vis_figs = []
    
    # Print DAGA status for debugging
    is_daga = getattr(actual_base_model, "use_daga", False)
    if is_daga:
        print(f"\n[DEBUG] Detection DAGA Visualization at epoch {epoch+1}:")
        print(f"  DAGA layers: {actual_base_model.vit_wrapper.daga_layers}")
        print(f"  Visualization layer: {actual_base_model.vit_wrapper.daga_guidance_layer_idx}")
        print(f"  Mix weights:")
        for layer_idx, daga_module in actual_base_model.vit_wrapper.daga_modules.items():
            weight_val = daga_module.mix_weight.item()
            print(f"    Layer {layer_idx}: {weight_val:.6f}")
        print()
    
    with torch.no_grad():
        # Get predictions and attention
        _, H, W = fixed_images[0].shape
        cls_logits, box_preds, centerness, adapted_attn_weights, guidance_map = base_model(fixed_images, True)
        
        detections = decode_predictions(
            cls_logits, box_preds, centerness,
            image_size=(H, W),
            stride=actual_base_model.stride,
            score_threshold=0.5,  # Only show high-confidence predictions
            nms_threshold=0.5,
            max_detections=30
        )
        
        # Process attention maps (only for DAGA models)
        adapted_attn_np = None
        baseline_attn_np = None
        
        if is_daga and adapted_attn_weights is not None:
            _, _, input_h, input_w = fixed_images.shape
            feat_h, feat_w = input_h // actual_base_model.stride, input_w // actual_base_model.stride
            num_patches = feat_h * feat_w
            adapted_attn_np = process_attention_weights(adapted_attn_weights, num_patches, feat_h, feat_w)
            
            # Get baseline attention from frozen backbone (no DAGA applied)
            # We extract attention at the same layer to have a fair comparison
            x_proc, _ = actual_base_model.vit_wrapper.vit.prepare_tokens_with_masks(fixed_images)
            baseline_raw_weights = None
            for i in range(actual_base_model.vit_wrapper.daga_guidance_layer_idx + 1):
                rope_sincos = (
                    actual_base_model.vit_wrapper.vit.rope_embed(H=feat_h, W=feat_w)
                    if actual_base_model.vit_wrapper.vit.rope_embed
                    else None
                )
                if i == actual_base_model.vit_wrapper.daga_guidance_layer_idx:
                    baseline_raw_weights = get_attention_map(actual_base_model.vit_wrapper.vit.blocks[i], x_proc)
                x_proc = actual_base_model.vit_wrapper.vit.blocks[i](x_proc, rope_sincos)
            
            if baseline_raw_weights is not None:
                baseline_attn_np = process_attention_weights(baseline_raw_weights, num_patches, feat_h, feat_w)
        
        images_np = fixed_images.cpu().numpy()
        vis_save_path = Path(output_dir) / "visualizations"
        vis_save_path.mkdir(parents=True, exist_ok=True)
        
        for j in range(images_np.shape[0]):
            # Denormalize image
            img = images_np[j].transpose(1, 2, 0)
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            img_h, img_w = img.shape[0], img.shape[1]
            
            # Group 1: Detection Results (GT vs Predictions)
            fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))
            mAP_val = val_metrics.get('mAP@50', 0.0) if val_metrics else 0.0
            fig1.suptitle(f"Detection Results - Epoch {epoch+1} - Sample {j} - mAP@50={mAP_val:.3f}", 
                        fontsize=14, fontweight="bold")
            
            # Left: Ground Truth
            axes1[0].imshow(img)
            num_gt_boxes = 0
            if j < len(fixed_boxes_list):
                boxes = fixed_boxes_list[j]
                
                # Handle both dict (with 'boxes' key) and direct tensor format
                if isinstance(boxes, dict):
                    if 'boxes' in boxes:
                        boxes = boxes['boxes']
                    else:
                        print(f"  [WARNING] boxes is dict but has no 'boxes' key: {list(boxes.keys())}")
                        boxes = None
                
                if boxes is not None:
                    if torch.is_tensor(boxes):
                        boxes = boxes.cpu().numpy()
                    
                    if len(boxes) > 0:
                        for box in boxes:
                            if len(box) == 4:
                                x1, y1, x2, y2 = box
                                rect = patches.Rectangle(
                                    (x1 * img_w, y1 * img_h),
                                    (x2 - x1) * img_w,
                                    (y2 - y1) * img_h,
                                    linewidth=3, edgecolor='lime', facecolor='none', alpha=1.0
                                )
                                axes1[0].add_patch(rect)
                                num_gt_boxes += 1
            axes1[0].set_title(f"Ground Truth ({num_gt_boxes} boxes)")
            axes1[0].axis("off")
            
            # Right: Predictions
            axes1[1].imshow(img)
            det = detections[j]
            pred_boxes = det['boxes'].cpu().numpy()
            pred_scores = det['scores'].cpu().numpy()
            
            for box, score in zip(pred_boxes, pred_scores):
                x1, y1, x2, y2 = box
                if score < 0.3:
                    color = 'red'
                    alpha = 0.6
                elif score < 0.5:
                    color = 'orange'
                    alpha = 0.8
                else:
                    color = 'lime'
                    alpha = 1.0
                
                rect = patches.Rectangle(
                    (x1 * img_w, y1 * img_h),
                    (x2 - x1) * img_w,
                    (y2 - y1) * img_h,
                    linewidth=3, edgecolor=color, facecolor='none', alpha=alpha
                )
                axes1[1].add_patch(rect)
                
                axes1[1].text(x1 * img_w, y1 * img_h - 8, 
                           f'{score:.2f}',
                           color='white', fontsize=10, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
            
            axes1[1].set_title(f"Predictions ({len(pred_boxes)} boxes)")
            axes1[1].axis("off")
            
            plt.tight_layout()
            vis_figs.append(fig1)
            fig1.savefig(vis_save_path / f"epoch_{epoch+1}_sample_{j}_detection.png", dpi=100)
            plt.close(fig1)
            
            # Group 2: Enhanced Attention Map Comparison (only if DAGA is used)
            if adapted_attn_np is not None and baseline_attn_np is not None:
                fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
                fig2.suptitle(f"Attention Analysis - Epoch {epoch+1} - Sample {j}", 
                            fontsize=16, fontweight="bold")
                
                # Row 1: Original attention maps
                im0 = axes2[0, 0].imshow(baseline_attn_np[j], cmap="viridis", vmin=0, vmax=1)
                axes2[0, 0].set_title("Frozen Backbone Attention", fontsize=12)
                axes2[0, 0].axis("off")
                plt.colorbar(im0, ax=axes2[0, 0], fraction=0.046, pad=0.04)
                
                im1 = axes2[0, 1].imshow(adapted_attn_np[j], cmap="viridis", vmin=0, vmax=1)
                axes2[0, 1].set_title("DAGA-Adapted Attention", fontsize=12)
                axes2[0, 1].axis("off")
                plt.colorbar(im1, ax=axes2[0, 1], fraction=0.046, pad=0.04)
                
                # Row 2: Difference map and overlay
                diff_map = adapted_attn_np[j] - baseline_attn_np[j]
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
                from scipy.ndimage import zoom
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
                vis_figs.append(fig2)
                fig2.savefig(vis_save_path / f"epoch_{epoch+1}_sample_{j}_attention.png", dpi=100)
                plt.close(fig2)
    
    return vis_figs


def prepare_visualization_data(val_dataset, args, device):
    """Prepare fixed images for visualization"""
    from torch.utils.data import DataLoader
    print("📸 Preparing visualization data...")
    indices = list(range(min(args.num_vis_samples, len(val_dataset))))
    vis_subset = torch.utils.data.Subset(val_dataset, indices)
    vis_loader = DataLoader(vis_subset, batch_size=len(indices), shuffle=False, 
                           collate_fn=detection_collate_fn)
    fixed_images, fixed_boxes, fixed_labels = next(iter(vis_loader))
    
    # Convert list to stacked tensor
    fixed_images = torch.stack(fixed_images).to(device)
    print("✓ Visualization data loaded.")
    return fixed_images, fixed_boxes


def run_training_loop(model, train_loader, val_loader, optimizer, scheduler, device, args, 
                     output_dir, fixed_vis_images, fixed_vis_boxes, num_classes, rank=0, world_size=1):
    """Execute main training loop"""
    is_main_process = (rank == 0)
    best_loss = float('inf')
    best_map = 0.0
    val_metrics = {'loss': float('inf')}
    start_time = time.time()
    
    for epoch in range(args.epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, num_classes)
        
        # Always compute mAP for detection (important metric)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        if is_main_process:
            print(f"\n📈 Epoch {epoch+1}/{args.epochs} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | mAP@50: {val_metrics.get('mAP@50', 0.0):.4f}")
            print(f"   Time Elapsed: {elapsed_time/60:.1f}min")
        
        if is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics['loss'],
                "mAP@50": val_metrics.get('mAP@50', 0.0),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total_time_minutes": elapsed_time / 60,
            }
            
            if args.enable_visualization and fixed_vis_images is not None and (
                epoch % args.log_freq == 0 or epoch == args.epochs - 1
            ):
                print("📊 Generating detection visualizations...")
                vis_figs = visualize_detection_results(
                    model, fixed_vis_images, fixed_vis_boxes, args, output_dir, epoch, val_metrics
                )
                if vis_figs:
                    log_dict["detection_results"] = [swanlab.Image(fig) for fig in vis_figs]
            
            swanlab.log(log_dict, step=epoch + 1) if getattr(args, 'enable_swanlab', True) else None
            
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                save_path = output_dir / "best_model.pth"
                from core.utils import save_checkpoint
                save_checkpoint(model, optimizer, epoch, best_loss, args, save_path)
                print(f"   ✅ New best model saved! (Loss: {best_loss:.4f})")
    
    return best_loss, val_metrics, (time.time() - start_time) / 60


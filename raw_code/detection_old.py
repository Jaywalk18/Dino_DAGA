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
from core.detr_components import DETRHead, detr_loss, box_cxcywh_to_xyxy
from core.backbones import get_attention_map, compute_daga_guidance_map, process_attention_weights
from core.utils import get_base_model


class DetectionModel(nn.Module):
    def __init__(
        self,
        pretrained_vit,
        num_classes=80,
        use_daga=False,
        daga_layers=[11],
        layers_to_use=[2, 5, 8, 11],  # Multi-layer features like official DINOv3
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.num_classes = num_classes
        self.use_daga = use_daga
        self.daga_layers = daga_layers
        self.layers_to_use = layers_to_use
        self.feature_dim = self.vit.embed_dim
        self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
        
        self.num_storage_tokens = -1
        
        # Freeze backbone
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # DAGA modules if enabled
        if self.use_daga:
            self.daga_modules = nn.ModuleDict(
                {str(i): DAGA(feature_dim=self.feature_dim) for i in daga_layers}
            )
            for param in self.daga_modules.parameters():
                param.requires_grad = True
        
        # DETR-style detection head
        total_feature_dim = self.feature_dim * len(layers_to_use)
        self.detection_head = DETRHead(
            feature_dim=total_feature_dim,
            num_classes=num_classes,
            num_queries=100,  # Standard DETR uses 100 queries
            hidden_dim=256,
            nheads=8,
            num_decoder_layers=6
        )
        
        for param in self.detection_head.parameters():
            param.requires_grad = True
        
        print(
            f"✓ DetectionModel initialized (DETR-style):\n"
            f"  - Feature dim: {self.feature_dim} x {len(layers_to_use)} layers = {total_feature_dim}\n"
            f"  - Layers to use: {layers_to_use}\n"
            f"  - Num classes: {num_classes}\n"
            f"  - Num queries: 100\n"
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
        
        # Compute DAGA guidance map if enabled
        if self.use_daga:
            daga_guidance_map = compute_daga_guidance_map(
                self.vit, x_processed, H, W, self.daga_guidance_layer_idx
            )
        
        # Extract multi-layer features (similar to official DINOv3)
        intermediate_features = []
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            if request_visualization_maps and idx == self.daga_guidance_layer_idx:
                with torch.no_grad():
                    adapted_attn_weights = get_attention_map(block, x_processed)
            
            x_processed = block(x_processed, rope_sincos)
            
            # Apply DAGA if needed
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
            
            # Collect intermediate layer features
            if idx in self.layers_to_use:
                patch_features = x_processed[:, 1 + num_registers:, :]
                feat_spatial = patch_features.transpose(1, 2).reshape(B, C, H, W)
                intermediate_features.append(feat_spatial)
        
        # Concatenate multi-layer features
        multi_layer_features = torch.cat(intermediate_features, dim=1)
        
        # DETR head returns (pred_logits, pred_boxes)
        pred_logits, pred_boxes = self.detection_head(multi_layer_features)
        
        return pred_logits, pred_boxes, adapted_attn_weights, daga_guidance_map


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
    
    # Better learning rate scaling for detection
    lr_scaled = args.lr * (args.batch_size * torch.cuda.device_count()) / 16.0
    
    # Use different learning rates and weight decay for different parts
    param_groups = [{"params": head_params, "lr": lr_scaled, "weight_decay": args.weight_decay}]
    if daga_params:
        param_groups.append(
            {"params": daga_params, "lr": lr_scaled * 0.5, "weight_decay": args.weight_decay * 0.5}
        )
    
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
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
    
    return optimizer, scheduler


# Legacy functions removed - now using DETR components

def _generalized_box_iou_loss(pred_boxes, target_boxes):
    """
    Compute GIoU loss between predicted and target boxes.
    Args:
        pred_boxes: (N, 4) in format [x1, y1, x2, y2], normalized [0,1], already sigmoid
        target_boxes: (N, 4) in format [x1, y1, x2, y2], normalized [0,1]
    Returns:
        GIoU loss (scalar)
    """
    # Boxes are already normalized to [0,1] from sigmoid
    pred_boxes = pred_boxes.clamp(0, 1)
    target_boxes = target_boxes.clamp(0, 1)
    
    # Compute areas
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    
    # Intersection
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Union
    union_area = pred_area + target_area - inter_area + 1e-7
    
    # IoU
    iou = inter_area / union_area
    
    # Enclosing box
    enclosing_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclosing_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclosing_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclosing_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    
    enclosing_area = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1) + 1e-7
    
    # GIoU
    giou = iou - (enclosing_area - union_area) / enclosing_area
    
    # GIoU loss: 1 - GIoU, range [0, 2]
    # Clamp to avoid negative loss
    loss = torch.clamp(1.0 - giou, min=0.0, max=2.0)
    
    return loss.mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss for classification (handles class imbalance)
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    p_t = torch.exp(-ce_loss)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * ce_loss).mean()


def _old_detection_loss(cls_pred, bbox_pred, obj_pred, boxes_list, labels_list):
    """
    Simplified detection loss with CrossEntropy and L1 Loss.
    
    Args:
        cls_pred: (B, num_anchors * num_classes, H, W)
        bbox_pred: (B, num_anchors * 4, H, W) - already sigmoid normalized
        obj_pred: (B, num_anchors, H, W)
        boxes_list: list of (N, 4) tensors - GT boxes (normalized [0,1])
        labels_list: list of (N,) tensors - GT labels
    """
    device = cls_pred.device
    batch_size = cls_pred.size(0)
    num_anchors = 3
    num_classes = cls_pred.size(1) // num_anchors
    H, W = cls_pred.size(2), cls_pred.size(3)
    
    # Reshape predictions
    cls_pred = cls_pred.view(batch_size, num_anchors, num_classes, H, W)
    bbox_pred = bbox_pred.view(batch_size, num_anchors, 4, H, W)
    obj_pred = obj_pred.view(batch_size, num_anchors, H, W)
    
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_obj_loss = 0.0
    num_pos_samples = 0
    
    for b in range(batch_size):
        boxes = boxes_list[b]
        labels = labels_list[b]
        
        obj_target = torch.zeros(num_anchors, H, W, device=device)
        
        if len(boxes) > 0:
            # Convert normalized boxes to feature map scale
            boxes_fm = boxes.clone()
            boxes_fm[:, [0, 2]] *= W
            boxes_fm[:, [1, 3]] *= H
            
            # Center-based assignment
            cx = (boxes_fm[:, 0] + boxes_fm[:, 2]) / 2
            cy = (boxes_fm[:, 1] + boxes_fm[:, 3]) / 2
            
            for i in range(len(boxes)):
                label_i = int(labels[i].cpu().item())
                if label_i < 0 or label_i >= num_classes:
                    continue
                
                cx_int = int(cx[i].clamp(0, W-1).cpu().item())
                cy_int = int(cy[i].clamp(0, H-1).cpu().item())
                
                anchor_idx = 0  # Use first anchor
                
                obj_target[anchor_idx, cy_int, cx_int] = 1.0
                num_pos_samples += 1
                
                # Cross Entropy for classification
                pred_logits = cls_pred[b, anchor_idx, :, cy_int, cx_int].unsqueeze(0)
                total_cls_loss += F.cross_entropy(pred_logits, labels[i:i+1], reduction='mean')
                
                # L1 Loss for bbox (simpler and more stable than GIoU)
                pred_box = bbox_pred[b, anchor_idx, :, cy_int, cx_int]
                target_box = boxes[i]
                total_bbox_loss += F.l1_loss(pred_box, target_box, reduction='mean')
        
        # Binary cross entropy for objectness
        obj_pos = obj_pred[b][obj_target == 1]
        obj_neg = obj_pred[b][obj_target == 0]
        
        if len(obj_pos) > 0:
            pos_loss = F.binary_cross_entropy_with_logits(obj_pos, torch.ones_like(obj_pos), reduction='mean')
            total_obj_loss += pos_loss
        
        if len(obj_neg) > 0:
            neg_loss = F.binary_cross_entropy_with_logits(obj_neg, torch.zeros_like(obj_neg), reduction='mean')
            total_obj_loss += neg_loss * 0.1  # Reduced weight for negative samples
    
    # Normalize
    num_pos_samples = max(num_pos_samples, 1)
    cls_loss = total_cls_loss / num_pos_samples
    bbox_loss = total_bbox_loss / num_pos_samples
    obj_loss = total_obj_loss / batch_size
    
    # Ensure all losses are positive
    cls_loss = torch.abs(cls_loss)
    bbox_loss = torch.abs(bbox_loss)
    obj_loss = torch.abs(obj_loss)
    
    # Combined loss with balanced weights
    loss = cls_loss + 2.0 * bbox_loss + 0.5 * obj_loss
    
    return loss


def train_epoch(model, dataloader, optimizer, device, epoch, num_classes):
    """Train for one epoch with DETR loss"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    
    for batch_idx, (images, boxes_list, labels_list) in enumerate(pbar):
        images = images.to(device)
        # Move boxes and labels to device
        boxes_list = [boxes.to(device) for boxes in boxes_list]
        labels_list = [labels.to(device) for labels in labels_list]
        
        optimizer.zero_grad()
        
        # Forward pass - returns (pred_logits, pred_boxes)
        pred_logits, pred_boxes, _, _ = model(images, False)
        
        # DETR loss
        loss_dict = detr_loss(pred_logits, pred_boxes, boxes_list, labels_list, num_classes)
        loss = loss_dict['loss']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        running_avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({
            "Loss": f"{running_avg_loss:.4f}",
            "CE": f"{loss_dict['loss_ce'].item():.3f}",
            "BBox": f"{loss_dict['loss_bbox'].item():.3f}",
            "GIoU": f"{loss_dict['loss_giou'].item():.3f}"
        })
    
    return total_loss / len(dataloader)


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes (x1,y1,x2,y2 format)
    """
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[2], box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[2], box2[3]
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-6)


def decode_predictions_detr(pred_logits, pred_boxes, conf_threshold=0.3):
    """
    Decode DETR predictions to bounding boxes
    
    Args:
        pred_logits: (B, num_queries, num_classes)
        pred_boxes: (B, num_queries, 4) in cxcywh format, normalized
        conf_threshold: confidence threshold for filtering
    
    Returns:
        List of (boxes, scores, labels) for each image in batch
    """
    batch_size = pred_logits.size(0)
    
    results = []
    
    for b in range(batch_size):
        logits = pred_logits[b]  # (num_queries, num_classes)
        boxes = pred_boxes[b]  # (num_queries, 4)
        
        # Get class probabilities
        probs = logits.softmax(-1)  # (num_queries, num_classes)
        scores, labels = probs.max(-1)  # (num_queries,)
        
        # Filter by confidence
        keep = scores > conf_threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]
        
        # Convert boxes from cxcywh to xyxy format
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        
        results.append((boxes_xyxy.cpu(), scores.cpu(), labels.cpu()))
    
    return results


def simple_nms(boxes, scores, threshold):
    """Simple NMS implementation"""
    if len(boxes) == 0:
        return []
    
    # Sort by scores
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while sorted_indices:
        current = sorted_indices[0]
        keep.append(current)
        sorted_indices = sorted_indices[1:]
        
        if not sorted_indices:
            break
        
        # Remove overlapping boxes
        current_box = boxes[current]
        new_indices = []
        
        for idx in sorted_indices:
            iou = compute_iou(current_box, boxes[idx])
            if iou < threshold:
                new_indices.append(idx)
        
        sorted_indices = new_indices
    
    return keep


def compute_detection_metrics(pred_boxes_list, pred_labels_list, gt_boxes_list, gt_labels_list, iou_threshold=0.5):
    """
    Compute detection metrics (mAP-like metric).
    
    Args:
        pred_boxes_list: List of predicted boxes per image (each is list of tensors)
        pred_labels_list: List of predicted labels per image (each is list of ints)
        gt_boxes_list: List of GT boxes per image (each is tensor)
        gt_labels_list: List of GT labels per image (each is tensor)
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        Dictionary with precision, recall, f1, and mAP-like score
    """
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    for pred_boxes, pred_labels, gt_boxes, gt_labels in zip(
        pred_boxes_list, pred_labels_list, gt_boxes_list, gt_labels_list
    ):
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        
        total_gt += len(gt_boxes)
        
        if len(pred_boxes) == 0:
            continue
        
        # Track which GT boxes have been matched
        gt_matched = [False] * len(gt_boxes)
        
        # For each prediction
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching GT box with same label
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_matched[gt_idx]:
                    continue
                
                if pred_label == gt_label.item():
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Check if it's a true positive
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                total_tp += 1
                gt_matched[best_gt_idx] = True
            else:
                total_fp += 1
    
    # Compute metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_gt + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    # mAP-like score (simplified)
    map_score = (precision + recall) / 2
    
    return {
        'precision': precision * 100,  # Convert to percentage
        'recall': recall * 100,
        'f1': f1 * 100,
        'mAP': map_score * 100,
        'tp': total_tp,
        'fp': total_fp,
        'total_gt': total_gt
    }


def evaluate(model, dataloader, device, num_classes):
    """Evaluate detection model with DETR loss and mAP metrics"""
    model.eval()
    total_loss = 0.0
    
    all_pred_boxes = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    
    with torch.no_grad():
        for images, boxes_list, labels_list in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            boxes_list = [boxes.to(device) for boxes in boxes_list]
            labels_list = [labels.to(device) for labels in labels_list]
            
            pred_logits, pred_boxes, _, _ = model(images, False)
            
            # Compute loss
            loss_dict = detr_loss(pred_logits, pred_boxes, boxes_list, labels_list, num_classes)
            total_loss += loss_dict['loss'].item()
            
            # Decode predictions for metrics - lower threshold for evaluation
            decoded = decode_predictions_detr(pred_logits, pred_boxes, conf_threshold=0.05)
            
            for (pred_boxes, pred_scores, pred_labels), gt_boxes, gt_labels in zip(
                decoded, boxes_list, labels_list
            ):
                all_pred_boxes.append([pred_boxes[i].tolist() for i in range(len(pred_boxes))])
                all_pred_labels.append([pred_labels[i].item() for i in range(len(pred_labels))])
                all_gt_boxes.append(gt_boxes.cpu())
                all_gt_labels.append(gt_labels.cpu())
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_detection_metrics(
        all_pred_boxes, all_pred_labels, all_gt_boxes, all_gt_labels
    )
    metrics['loss'] = avg_loss
    
    return metrics


def visualize_detection_results(
    model, fixed_images, fixed_boxes_list, args, output_dir, epoch
):
    """Visualize detection predictions with attention maps (separated into two groups)"""
    if fixed_images is None:
        return []
    
    base_model = get_base_model(model)
    base_model.eval()
    vis_figs = []
    
    # Get actual model from DDP wrapper if needed
    from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
    actual_base_model = base_model.module if isinstance(base_model, (DataParallel, DDP)) else base_model
    
    with torch.no_grad():
        _, (H, W) = actual_base_model.vit_wrapper.vit.prepare_tokens_with_masks(fixed_images)
        num_patches_expected = H * W
        
        pred_logits, pred_boxes, adapted_attn_weights, _ = base_model(
            fixed_images, True
        )
        
        adapted_attn_np = None
        baseline_attn_np = None
        
        if adapted_attn_weights is not None:
            adapted_attn_np = process_attention_weights(adapted_attn_weights, num_patches_expected, H, W)
            
            x_proc, _ = actual_base_model.vit_wrapper.vit.prepare_tokens_with_masks(fixed_images)
            baseline_raw_weights = None
            for i in range(actual_base_model.vit_wrapper.daga_guidance_layer_idx + 1):
                rope_sincos = (
                    actual_base_model.vit_wrapper.vit.rope_embed(H=H, W=W)
                    if actual_base_model.vit_wrapper.vit.rope_embed
                    else None
                )
                if i == actual_base_model.vit_wrapper.daga_guidance_layer_idx:
                    baseline_raw_weights = get_attention_map(actual_base_model.vit_wrapper.vit.blocks[i], x_proc)
                x_proc = actual_base_model.vit_wrapper.vit.blocks[i](x_proc, rope_sincos)
            
            if baseline_raw_weights is not None:
                baseline_attn_np = process_attention_weights(baseline_raw_weights, num_patches_expected, H, W)
        
        images_np = fixed_images.cpu().numpy()
        
        vis_save_path = Path(output_dir) / "visualizations"
        vis_save_path.mkdir(parents=True, exist_ok=True)
        
        for j in range(images_np.shape[0]):
            # Group 1: Detection Results (GT vs Predictions side by side)
            fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))
            fig1.suptitle(f"Detection Results - Epoch {epoch+1} - Sample {j}", fontsize=14, fontweight="bold")
            
            img = images_np[j].transpose(1, 2, 0)
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            
            # Left: Ground Truth Boxes
            axes1[0].imshow(img)
            if j < len(fixed_boxes_list):
                boxes = fixed_boxes_list[j]
                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        rect = patches.Rectangle(
                            (x1*img.shape[1], y1*img.shape[0]), 
                            (x2-x1)*img.shape[1], 
                            (y2-y1)*img.shape[0],
                            linewidth=2, edgecolor='lime', facecolor='none', label='GT'
                        )
                        axes1[0].add_patch(rect)
            axes1[0].set_title("Ground Truth Boxes")
            axes1[0].axis("off")
            
            # Right: Predicted Boxes (DETR output)
            axes1[1].imshow(img)
            
            # Decode predictions
            logits_j = pred_logits[j:j+1]  # (1, num_queries, num_classes)
            boxes_j = pred_boxes[j:j+1]  # (1, num_queries, 4)
            
            # Debug: Print raw predictions
            if j == 0:
                print(f"\n[DEBUG] Sample {j} - Raw predictions:")
                print(f"  Box range: [{boxes_j[0].min().item():.3f}, {boxes_j[0].max().item():.3f}]")
                print(f"  Box mean: {boxes_j[0].mean().item():.3f}")
                print(f"  Sample boxes (first 5):")
                for idx in range(min(5, boxes_j.shape[1])):
                    print(f"    Query {idx}: cx={boxes_j[0,idx,0]:.3f}, cy={boxes_j[0,idx,1]:.3f}, w={boxes_j[0,idx,2]:.3f}, h={boxes_j[0,idx,3]:.3f}")
            
            # Lower threshold for visualization to see more predictions
            decoded = decode_predictions_detr(logits_j, boxes_j, conf_threshold=0.1)
            pred_boxes_vis, pred_scores_vis, pred_labels_vis = decoded[0]
            
            print(f"  Sample {j}: {len(pred_boxes_vis)} predictions above threshold")
            
            # Show top predictions
            for pred_box, score in zip(pred_boxes_vis[:10], pred_scores_vis[:10]):
                x1, y1, x2, y2 = pred_box.numpy()
                score_val = score.item()
                
                # Draw prediction box
                rect = patches.Rectangle(
                    (x1*img.shape[1], y1*img.shape[0]), 
                    (x2-x1)*img.shape[1], 
                    (y2-y1)*img.shape[0],
                    linewidth=2, edgecolor='red', facecolor='none', alpha=min(score_val, 1.0)
                )
                axes1[1].add_patch(rect)
                # Add score text
                axes1[1].text(x1*img.shape[1], y1*img.shape[0]-5, 
                             f'{score_val:.2f}', color='red', fontsize=8, 
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            axes1[1].set_title(f"Predicted Boxes (top {len(pred_boxes_vis[:10])} detections)")
            axes1[1].axis("off")
            
            plt.tight_layout()
            vis_figs.append(fig1)
            
            fig1.savefig(
                vis_save_path / f"epoch_{epoch+1}_sample_{j}_detection.png",
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
                
                plt.tight_layout()
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
    from data.detection_datasets import detection_collate_fn
    
    indices = list(range(min(args.num_vis_samples, len(val_dataset))))
    vis_subset = torch.utils.data.Subset(val_dataset, indices)
    vis_loader = DataLoader(vis_subset, batch_size=len(indices), shuffle=False, 
                           collate_fn=detection_collate_fn)
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
    num_classes,
    rank=0,
    world_size=1,
):
    """Execute main training and evaluation loop with DETR"""
    is_main_process = (rank == 0)
    best_loss = float('inf')
    best_map = 0.0
    val_metrics = {'loss': float('inf')}
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, num_classes)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        # Only print on main process
        if is_main_process:
            print(f"\n📈 Epoch {epoch+1}/{args.epochs} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"   mAP: {val_metrics['mAP']:.2f}% | Precision: {val_metrics['precision']:.2f}% | Recall: {val_metrics['recall']:.2f}%")
            print(f"   TP: {val_metrics['tp']} | FP: {val_metrics['fp']} | GT: {val_metrics['total_gt']}")
            print(f"   Time Elapsed: {elapsed_time/60:.1f}min")
        
        if is_main_process:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics['loss'],
                "val_mAP": val_metrics['mAP'],
                "val_precision": val_metrics['precision'],
                "val_recall": val_metrics['recall'],
                "val_f1": val_metrics['f1'],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total_time_minutes": elapsed_time / 60,
            }
            
            if args.enable_visualization and fixed_vis_images is not None and (
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
            
            swanlab.log(log_dict, step=epoch + 1) if getattr(args, 'enable_swanlab', True) else None
            
            # Save best model based on mAP
            if val_metrics['mAP'] > best_map:
                best_map = val_metrics['mAP']
                best_loss = val_metrics['loss']
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
                print(f"   ✅ New best model saved! (mAP: {best_map:.2f}%, Loss: {best_loss:.4f})")
    
    return best_loss, val_metrics, (time.time() - start_time) / 60

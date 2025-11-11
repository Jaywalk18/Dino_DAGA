"""
Simple detection head for quick DAGA testing
Based on FCOS (center-based detection)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDetectionHead(nn.Module):
    """Simple FCOS-style detection head for visualization"""
    
    def __init__(self, in_channels, num_classes, num_conv=3):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared conv layers
        conv_layers = []
        for i in range(num_conv):
            conv_layers.extend([
                nn.Conv2d(in_channels if i == 0 else 256, 256, 3, padding=1),
                nn.GroupNorm(32, 256),
                nn.ReLU(inplace=True)
            ])
        self.conv = nn.Sequential(*conv_layers)
        
        # Classification branch
        self.cls_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.cls_head = nn.Conv2d(256, num_classes, 1)
        
        # Box regression branch (l, t, r, b)
        self.box_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.box_head = nn.Conv2d(256, 4, 1)
        
        # Centerness branch
        self.ctr_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.ctr_head = nn.Conv2d(256, 1, 1)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Prior for classification (better initialization)
        nn.init.constant_(self.cls_head.bias, -4.6)  # ~0.01 probability
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            cls_logits: (B, num_classes, H, W)
            box_preds: (B, 4, H, W) - (l, t, r, b) distances
            centerness: (B, 1, H, W)
        """
        feat = self.conv(x)
        
        # Classification
        cls_feat = F.relu(self.cls_conv(feat))
        cls_logits = self.cls_head(cls_feat)
        
        # Box regression
        box_feat = F.relu(self.box_conv(feat))
        box_preds = F.relu(self.box_head(box_feat))  # Ensure positive distances
        
        # Centerness
        ctr_feat = F.relu(self.ctr_conv(feat))
        centerness = self.ctr_head(ctr_feat)
        
        return cls_logits, box_preds, centerness


def compute_locations(h, w, stride, device):
    """Compute center locations for each feature map position"""
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def simple_detection_loss(cls_logits, box_preds, centerness, 
                          gt_boxes_list, gt_labels_list, 
                          image_size, stride=16):
    """
    Simple detection loss
    
    Args:
        cls_logits: (B, num_classes, H, W)
        box_preds: (B, 4, H, W)
        centerness: (B, 1, H, W)
        gt_boxes_list: list of (N, 4) tensors in xyxy format (normalized)
        gt_labels_list: list of (N,) tensors
        image_size: (H, W) of input image
        stride: feature map stride
    """
    device = cls_logits.device
    B, C, H, W = cls_logits.shape
    
    # Reshape predictions
    cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
    box_preds = box_preds.permute(0, 2, 3, 1).reshape(-1, 4)  # (B*H*W, 4)
    centerness = centerness.permute(0, 2, 3, 1).reshape(-1)  # (B*H*W,)
    
    # Compute locations
    locations = compute_locations(H, W, stride, device)  # (H*W, 2)
    
    total_cls_loss = 0.0
    total_box_loss = 0.0
    total_ctr_loss = 0.0
    num_pos = 0
    
    for b in range(B):
        gt_boxes = gt_boxes_list[b]  # (N, 4) normalized xyxy
        gt_labels = gt_labels_list[b]  # (N,)
        
        if len(gt_boxes) == 0:
            continue
        
        # Convert gt boxes to pixel coordinates
        img_h, img_w = image_size
        gt_boxes_pixel = gt_boxes.clone()
        gt_boxes_pixel[:, [0, 2]] *= img_w
        gt_boxes_pixel[:, [1, 3]] *= img_h
        
        # Get predictions for this batch item
        start_idx = b * H * W
        end_idx = (b + 1) * H * W
        cls_logits_b = cls_logits[start_idx:end_idx]  # (H*W, C)
        box_preds_b = box_preds[start_idx:end_idx]  # (H*W, 4)
        centerness_b = centerness[start_idx:end_idx]  # (H*W,)
        
        # Assign targets
        labels, box_targets, ctr_targets = assign_targets(
            locations, gt_boxes_pixel, gt_labels, stride
        )
        
        # Classification loss (focal loss)
        pos_mask = labels >= 0
        num_pos_b = pos_mask.sum()
        
        if num_pos_b > 0:
            # Focal loss
            cls_targets = torch.zeros_like(cls_logits_b)
            cls_targets[pos_mask, labels[pos_mask]] = 1.0
            
            p = torch.sigmoid(cls_logits_b)
            focal_weight = (cls_targets - p).abs() ** 2
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_logits_b, cls_targets, reduction='none'
            )
            cls_loss = (focal_weight * cls_loss).sum() / num_pos_b
            
            # Box loss (IoU loss)
            box_preds_pos = box_preds_b[pos_mask]
            box_targets_pos = box_targets[pos_mask]
            locations_pos = locations[pos_mask]
            
            # Convert to xyxy for IoU computation
            pred_boxes = ltrb_to_xyxy(locations_pos, box_preds_pos)
            target_boxes = ltrb_to_xyxy(locations_pos, box_targets_pos)
            
            box_loss = 1 - compute_iou(pred_boxes, target_boxes).mean()
            
            # Centerness loss
            ctr_targets_pos = ctr_targets[pos_mask]
            ctr_loss = F.binary_cross_entropy_with_logits(
                centerness_b[pos_mask], ctr_targets_pos, reduction='mean'
            )
            
            total_cls_loss += cls_loss
            total_box_loss += box_loss
            total_ctr_loss += ctr_loss
            num_pos += num_pos_b
    
    # Average over batch
    if num_pos > 0:
        avg_cls_loss = total_cls_loss / B
        avg_box_loss = total_box_loss / B * 2.0  # Weight box loss more
        avg_ctr_loss = total_ctr_loss / B
    else:
        # Ensure gradient flow even when no positive samples
        # Use a small dummy loss to touch all parameters
        avg_cls_loss = (cls_logits.sum() + box_preds.sum() + centerness.sum()) * 0.0
        avg_box_loss = torch.tensor(0.0, device=cls_logits.device, requires_grad=True)
        avg_ctr_loss = torch.tensor(0.0, device=cls_logits.device, requires_grad=True)
    
    total_loss = avg_cls_loss + avg_box_loss + avg_ctr_loss
    
    return {
        'loss': total_loss,
        'cls_loss': avg_cls_loss,
        'box_loss': avg_box_loss,
        'ctr_loss': avg_ctr_loss
    }


def assign_targets(locations, gt_boxes, gt_labels, stride):
    """
    Assign GT boxes to locations
    
    Args:
        locations: (H*W, 2) center coordinates
        gt_boxes: (N, 4) in xyxy format (pixel)
        gt_labels: (N,)
        stride: feature stride
    Returns:
        labels: (H*W,) class labels (-1 for background)
        box_targets: (H*W, 4) ltrb distances
        ctr_targets: (H*W,) centerness values
    """
    num_loc = locations.shape[0]
    num_gt = gt_boxes.shape[0]
    device = locations.device
    
    # Expand for broadcasting
    locations_expanded = locations[:, None, :].expand(num_loc, num_gt, 2)  # (H*W, N, 2)
    gt_boxes_expanded = gt_boxes[None, :, :].expand(num_loc, num_gt, 4)  # (H*W, N, 4)
    
    # Compute distances (l, t, r, b)
    l = locations_expanded[:, :, 0] - gt_boxes_expanded[:, :, 0]
    t = locations_expanded[:, :, 1] - gt_boxes_expanded[:, :, 1]
    r = gt_boxes_expanded[:, :, 2] - locations_expanded[:, :, 0]
    b = gt_boxes_expanded[:, :, 3] - locations_expanded[:, :, 1]
    
    ltrb = torch.stack([l, t, r, b], dim=2)  # (H*W, N, 4)
    
    # Check if location is inside box
    inside_mask = (ltrb.min(dim=2)[0] > 0)  # (H*W, N)
    
    # Find closest GT for each location
    distances = ltrb.sum(dim=2)  # (H*W, N)
    distances[~inside_mask] = float('inf')
    
    min_dist, min_idx = distances.min(dim=1)  # (H*W,)
    
    # Initialize targets
    labels = torch.full((num_loc,), -1, dtype=torch.long, device=device)
    box_targets = torch.zeros((num_loc, 4), device=device)
    ctr_targets = torch.zeros(num_loc, device=device)
    
    # Assign positive samples
    pos_mask = min_dist < float('inf')
    
    if pos_mask.sum() > 0:
        labels[pos_mask] = gt_labels[min_idx[pos_mask]]
        box_targets[pos_mask] = ltrb[pos_mask, min_idx[pos_mask]]
        
        # Compute centerness
        l_pos = box_targets[pos_mask, 0]
        t_pos = box_targets[pos_mask, 1]
        r_pos = box_targets[pos_mask, 2]
        b_pos = box_targets[pos_mask, 3]
        
        centerness_val = torch.sqrt(
            (torch.min(l_pos, r_pos) / torch.max(l_pos, r_pos)) *
            (torch.min(t_pos, b_pos) / torch.max(t_pos, b_pos))
        )
        ctr_targets[pos_mask] = centerness_val
    
    return labels, box_targets, ctr_targets


def ltrb_to_xyxy(locations, ltrb):
    """Convert (l, t, r, b) to (x1, y1, x2, y2)"""
    x1 = locations[:, 0] - ltrb[:, 0]
    y1 = locations[:, 1] - ltrb[:, 1]
    x2 = locations[:, 0] + ltrb[:, 2]
    y2 = locations[:, 1] + ltrb[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=1)


def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes (for mAP calculation).
    
    Args:
        boxes1: (N, 4) tensor in xyxy format
        boxes2: (M, 4) tensor in xyxy format
    
    Returns:
        iou: (N, M) tensor
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou


def decode_predictions(cls_logits, box_preds, centerness, image_size, 
                      stride=16, score_threshold=0.15, nms_threshold=0.6, max_detections=30):
    """
    Decode predictions to boxes
    
    Args:
        cls_logits: (B, num_classes, H, W)
        box_preds: (B, 4, H, W)
        centerness: (B, 1, H, W)
        image_size: (H, W)
        
    Returns:
        List of detections, each containing:
            boxes: (N, 4) in xyxy format (normalized)
            scores: (N,)
            labels: (N,)
    """
    device = cls_logits.device
    B, C, H, W = cls_logits.shape
    img_h, img_w = image_size
    
    # Compute locations
    locations = compute_locations(H, W, stride, device)  # (H*W, 2)
    
    # Reshape
    cls_logits = cls_logits.permute(0, 2, 3, 1)  # (B, H, W, C)
    box_preds = box_preds.permute(0, 2, 3, 1)  # (B, H, W, 4)
    centerness = centerness.permute(0, 2, 3, 1)  # (B, H, W, 1)
    
    all_detections = []
    
    for b in range(B):
        cls_scores = torch.sigmoid(cls_logits[b]).reshape(-1, C)  # (H*W, C)
        box_pred = box_preds[b].reshape(-1, 4)  # (H*W, 4)
        ctr = torch.sigmoid(centerness[b]).reshape(-1)  # (H*W,)
        
        # Multiply by centerness for better scoring
        cls_scores = cls_scores * ctr[:, None] ** 0.5  # Less aggressive centerness weighting
        
        # Get top predictions
        max_scores, max_labels = cls_scores.max(dim=1)
        keep = max_scores > score_threshold
        
        if keep.sum() == 0:
            all_detections.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros(0, device=device),
                'labels': torch.zeros(0, dtype=torch.long, device=device)
            })
            continue
        
        # Filter
        scores_keep = max_scores[keep]
        labels_keep = max_labels[keep]
        box_pred_keep = box_pred[keep]
        locations_keep = locations[keep]
        
        # Convert to xyxy (pixel coordinates)
        boxes_pixel = ltrb_to_xyxy(locations_keep, box_pred_keep)
        
        # Normalize to [0, 1]
        boxes_norm = boxes_pixel.clone()
        boxes_norm[:, [0, 2]] /= img_w
        boxes_norm[:, [1, 3]] /= img_h
        boxes_norm = boxes_norm.clamp(0, 1)
        
        # NMS per class
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for cls_id in labels_keep.unique():
            cls_mask = labels_keep == cls_id
            cls_boxes = boxes_pixel[cls_mask]
            cls_scores = scores_keep[cls_mask]
            cls_boxes_norm = boxes_norm[cls_mask]
            
            # Simple NMS
            keep_nms = torchvision_nms(cls_boxes, cls_scores, nms_threshold)
            
            final_boxes.append(cls_boxes_norm[keep_nms])
            final_scores.append(cls_scores[keep_nms])
            final_labels.append(labels_keep[cls_mask][keep_nms])
        
        if len(final_boxes) > 0:
            final_boxes = torch.cat(final_boxes, dim=0)
            final_scores = torch.cat(final_scores, dim=0)
            final_labels = torch.cat(final_labels, dim=0)
            
            # Keep top K
            if len(final_scores) > max_detections:
                top_k = torch.topk(final_scores, max_detections)[1]
                final_boxes = final_boxes[top_k]
                final_scores = final_scores[top_k]
                final_labels = final_labels[top_k]
        else:
            final_boxes = torch.zeros((0, 4), device=device)
            final_scores = torch.zeros(0, device=device)
            final_labels = torch.zeros(0, dtype=torch.long, device=device)
        
        all_detections.append({
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        })
    
    return all_detections


def torchvision_nms(boxes, scores, iou_threshold):
    """Simple NMS implementation"""
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0].item()
        keep.append(i)
        
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


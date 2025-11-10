"""
DETR Components for Object Detection
Simplified but complete implementation based on official DINOv3 detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class MLP(nn.Module):
    """Simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer with self-attention and cross-attention"""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        """
        Args:
            tgt: query features (num_queries, batch, d_model)
            memory: encoder features (H*W, batch, d_model)
        """
        # Self-attention
        q = k = tgt
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2 = self.cross_attn(query=tgt, key=memory, value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerDecoder(nn.Module):
    """Transformer decoder with multiple layers"""
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                decoder_layer.self_attn.embed_dim,
                decoder_layer.self_attn.num_heads,
                decoder_layer.linear1.out_features,
                decoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory)
        return output


class DETRHead(nn.Module):
    """
    DETR-style detection head
    Uses learnable queries and transformer decoder for object detection
    """
    def __init__(self, feature_dim, num_classes, num_queries=100, hidden_dim=256, 
                 nheads=8, num_decoder_layers=6):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Project input features to hidden dimension
        self.input_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        
        # Learnable query embeddings - split into content and positional
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        
        # Reference points for queries (helps with diversity)
        self.reference_points = nn.Linear(hidden_dim, 2)
        
        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Initialize
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        
        # Initialize query embeddings with more diversity
        # Use Xavier for better initial diversity
        nn.init.xavier_uniform_(self.query_embed.weight)
        
        # Initialize reference points
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0)
        
        # Initialize class head with prior
        prior_prob = 0.01
        bias_value = -(torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        nn.init.constant_(self.class_embed.bias, bias_value)
        
        # Initialize bbox head - crucial for spreading predictions
        # Use uniform distribution for better initial diversity
        for layer in self.bbox_embed.layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        
        # Last layer: Initialize to predict diverse box positions
        # Use uniform initialization to ensure diversity
        last_layer = self.bbox_embed.layers[-1]
        nn.init.uniform_(last_layer.weight, -0.1, 0.1)
        
        # Initialize bias to encourage diverse predictions before sigmoid
        # sigmoid(-2) ≈ 0.12, sigmoid(0) = 0.5, sigmoid(2) ≈ 0.88
        # We want initial predictions spread across the image
        with torch.no_grad():
            last_layer.bias[0:2].uniform_(-1, 1)  # cx, cy: spread across image
            last_layer.bias[2:4].uniform_(-0.5, 0.5)  # w, h: varied sizes
    
    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) - concatenated multi-scale features
        Returns:
            pred_logits: (B, num_queries, num_classes)
            pred_boxes: (B, num_queries, 4) - normalized [cx, cy, w, h]
        """
        B, C, H, W = features.shape
        
        # Project features
        features = self.input_proj(features)  # (B, hidden_dim, H, W)
        
        # Flatten spatial dimensions for transformer
        # (B, hidden_dim, H, W) -> (H*W, B, hidden_dim)
        memory = features.flatten(2).permute(2, 0, 1)
        
        # Get query embeddings - split content and position
        # (num_queries, hidden_dim * 2) -> (num_queries, B, hidden_dim * 2)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        query_pos, query_content = torch.split(query_embed, self.hidden_dim, dim=2)
        
        # Pass through transformer decoder with positional info
        # Initialize tgt with content queries (not zeros)
        tgt = query_content
        hs = self.transformer_decoder(tgt + query_pos, memory)  # (num_queries, B, hidden_dim)
        
        # Transpose to (B, num_queries, hidden_dim)
        hs = hs.transpose(0, 1)
        
        # Predict class and bbox for each query
        pred_logits = self.class_embed(hs)  # (B, num_queries, num_classes)
        pred_boxes_raw = self.bbox_embed(hs)  # (B, num_queries, 4)
        
        # Apply sigmoid to ensure boxes are in [0, 1]
        # But first add a small offset to avoid all-zero gradients
        pred_boxes = pred_boxes_raw.sigmoid()  # (B, num_queries, 4)
        
        return pred_logits, pred_boxes


def box_cxcywh_to_xyxy(boxes):
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes):
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Compute generalized IoU between two sets of boxes
    boxes1, boxes2: (N, 4) and (M, 4) in xyxy format
    Returns: (N, M) matrix of GIoU values
    """
    # Ensure boxes are valid
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Union
    union = area1[:, None] + area2 - inter
    iou = inter / union
    
    # Enclosing box
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]
    
    # GIoU
    giou = iou - (area_enc - union) / area_enc
    return giou


@torch.no_grad()
def hungarian_matcher(pred_logits, pred_boxes, gt_labels, gt_boxes, cost_class=1, cost_bbox=5, cost_giou=2):
    """
    Perform Hungarian matching between predictions and ground truth
    
    Args:
        pred_logits: (num_queries, num_classes)
        pred_boxes: (num_queries, 4) in cxcywh format, normalized
        gt_labels: (num_gt,) - ground truth class labels
        gt_boxes: (num_gt, 4) in xyxy format, normalized
        
    Returns:
        matched_indices: list of (pred_idx, gt_idx) pairs
    """
    num_queries = pred_logits.shape[0]
    num_gt = len(gt_labels)
    
    if num_gt == 0:
        return [], []
    
    # Classification cost - use softmax probabilities
    pred_probs = pred_logits.softmax(-1)  # (num_queries, num_classes)
    cost_class_matrix = -pred_probs[:, gt_labels]  # (num_queries, num_gt)
    
    # Convert pred_boxes to xyxy for IoU computation
    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)  # (num_queries, 4)
    
    # L1 cost on boxes
    cost_bbox_matrix = torch.cdist(pred_boxes, box_xyxy_to_cxcywh(gt_boxes), p=1)
    
    # GIoU cost
    cost_giou_matrix = -generalized_box_iou(pred_boxes_xyxy, gt_boxes)
    
    # Final cost matrix
    cost_matrix = cost_class * cost_class_matrix + cost_bbox * cost_bbox_matrix + cost_giou * cost_giou_matrix
    
    # Hungarian algorithm
    cost_matrix = cost_matrix.cpu().numpy()
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    return pred_indices.tolist(), gt_indices.tolist()


def detr_loss(pred_logits, pred_boxes, gt_boxes_list, gt_labels_list, 
              num_classes, cost_class=1, cost_bbox=5, cost_giou=2, 
              eos_coef=0.1, bbox_loss_coef=5.0, giou_loss_coef=2.0):
    """
    Compute DETR loss with Hungarian matching
    
    Args:
        pred_logits: (B, num_queries, num_classes)
        pred_boxes: (B, num_queries, 4) in cxcywh format, normalized
        gt_boxes_list: list of (N, 4) tensors in xyxy format, normalized
        gt_labels_list: list of (N,) tensors with class labels
        num_classes: number of object classes
        eos_coef: weight for no-object class
        
    Returns:
        loss_dict: dictionary with loss components
    """
    device = pred_logits.device
    batch_size = pred_logits.shape[0]
    num_queries = pred_logits.shape[1]
    
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_giou_loss = 0.0
    num_boxes = 0
    
    for b in range(batch_size):
        pred_logits_b = pred_logits[b]  # (num_queries, num_classes)
        pred_boxes_b = pred_boxes[b]  # (num_queries, 4)
        gt_boxes_b = gt_boxes_list[b]  # (num_gt, 4)
        gt_labels_b = gt_labels_list[b]  # (num_gt,)
        
        # Hungarian matching
        pred_indices, gt_indices = hungarian_matcher(
            pred_logits_b, pred_boxes_b, gt_labels_b, gt_boxes_b,
            cost_class, cost_bbox, cost_giou
        )
        
        num_boxes += len(gt_labels_b)
        
        # Classification loss
        # DETR uses num_classes as background, so expand logits
        pred_logits_expanded = torch.cat([
            pred_logits_b, 
            torch.zeros(num_queries, 1, device=device)
        ], dim=-1)  # (num_queries, num_classes + 1)
        
        target_classes = torch.full(
            (num_queries,), num_classes, dtype=torch.int64, device=device
        )  # Default to no-object class (index num_classes)
        target_classes[pred_indices] = gt_labels_b[gt_indices]
        
        # Weighted cross-entropy (down-weight no-object class)
        class_weights = torch.ones(num_classes + 1, device=device)
        class_weights[-1] = eos_coef
        cls_loss = F.cross_entropy(pred_logits_expanded, target_classes, weight=class_weights)
        total_cls_loss += cls_loss
        
        # Box losses (only for matched queries)
        if len(pred_indices) > 0:
            pred_boxes_matched = pred_boxes_b[pred_indices]
            gt_boxes_matched = gt_boxes_b[gt_indices]
            gt_boxes_matched_cxcywh = box_xyxy_to_cxcywh(gt_boxes_matched)
            
            # L1 loss
            bbox_loss = F.l1_loss(pred_boxes_matched, gt_boxes_matched_cxcywh, reduction='sum')
            total_bbox_loss += bbox_loss
            
            # GIoU loss
            pred_boxes_matched_xyxy = box_cxcywh_to_xyxy(pred_boxes_matched)
            giou_matrix = generalized_box_iou(pred_boxes_matched_xyxy, gt_boxes_matched)
            giou_loss = (1 - giou_matrix.diag()).sum()
            total_giou_loss += giou_loss
    
    # Normalize
    num_boxes = max(num_boxes, 1)
    cls_loss = total_cls_loss / batch_size
    bbox_loss = total_bbox_loss / num_boxes
    giou_loss = total_giou_loss / num_boxes
    
    # Combined loss with configurable coefficients
    loss = cls_loss + bbox_loss_coef * bbox_loss + giou_loss_coef * giou_loss
    
    return {
        'loss': loss,
        'loss_ce': cls_loss,
        'loss_bbox': bbox_loss,
        'loss_giou': giou_loss
    }


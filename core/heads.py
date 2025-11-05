import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Linear classification head for image classification"""
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        # Input: [CLS token + patch mean], so feature_dim * 2
        self.classifier = nn.Linear(feature_dim * 2, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)
        
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, features):
        return self.classifier(features)

class LinearSegmentationHead(nn.Module):
    """
    Lightweight segmentation head following official DINOv3 design.
    Dropout -> BN -> Conv1x1, extremely simple yet effective.
    """
    def __init__(self, in_channels, num_classes, use_bn=True):
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        self.num_classes = num_classes
        
        # Official DINOv3 uses minimal architecture:
        # Dropout -> BatchNorm -> 1x1 Conv
        self.dropout = nn.Dropout2d(0.1)
        self.bn = nn.SyncBatchNorm(self.channels) if use_bn else nn.Identity()
        self.conv = nn.Conv2d(self.channels, num_classes, kernel_size=1)
        
        # Initialize like official implementation
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)
    
    def _transform_inputs(self, features_list):
        """Transform and align multi-level features to same spatial size"""
        # Use first feature's spatial size as base
        base_size = features_list[0].shape[2:]
        
        # Interpolate all features to base size
        aligned_features = []
        for feat in features_list:
            if feat.shape[2:] != base_size:
                feat = F.interpolate(feat, size=base_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # Concatenate along channel dimension
        x = torch.cat(aligned_features, dim=1)
        return x
    
    def forward(self, features_list, target_size):
        # Transform and concatenate multi-scale features
        x = self._transform_inputs(features_list)
        
        # Simple pipeline following official: Dropout -> BN -> Conv
        x = self.dropout(x)
        x = self.bn(x)
        x = self.conv(x)
        
        # Ensure target_size is tuple of ints for interpolation
        if isinstance(target_size, torch.Tensor):
            target_size = tuple(target_size.tolist())
        elif not isinstance(target_size, (tuple, list)):
            target_size = (target_size, target_size)
        target_size = (int(target_size[0]), int(target_size[1]))
        
        # Upsample to target size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


class DetectionHead(nn.Module):
    """
    Detection head with improved architecture.
    Outputs: class logits, bbox coordinates, objectness scores
    Uses residual connections and layer normalization for better stability
    """
    def __init__(self, feature_dim, num_classes, num_anchors=3, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Feature projection
        self.proj = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim * 2, kernel_size=1),
            nn.GroupNorm(32, hidden_dim * 2),
            nn.ReLU(inplace=True)
        )
        
        # Feature refinement with residual blocks
        self.conv1 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(32, hidden_dim * 2)
        
        self.conv2 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(32, hidden_dim)
        
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(32, hidden_dim)
        
        # Prediction heads with separate feature processing
        self.cls_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(hidden_dim, num_anchors * num_classes, kernel_size=1)
        
        self.bbox_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bbox_head = nn.Conv2d(hidden_dim, num_anchors * 4, kernel_size=1)
        
        self.obj_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.obj_head = nn.Conv2d(hidden_dim, num_anchors, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize prediction heads with smaller std
        nn.init.normal_(self.cls_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_head.bias, 0)
        
        nn.init.normal_(self.bbox_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.bbox_head.bias, 0.5)  # Center bias
        
        # Objectness head with negative bias (sparse detection)
        nn.init.normal_(self.obj_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.obj_head.bias, -2.0)  # sigmoid(-2) ≈ 0.12
    
    def forward(self, x):
        # Initial projection
        x = self.proj(x)
        
        # Feature refinement with residual
        identity = x
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.gn2(self.conv2(x)))
        
        x = self.relu(self.gn3(self.conv3(x)))
        
        # Separate paths for different predictions
        cls_feat = self.relu(self.cls_conv(x))
        cls_output = self.cls_head(cls_feat)
        
        bbox_feat = self.relu(self.bbox_conv(x))
        bbox_output = torch.sigmoid(self.bbox_head(bbox_feat))  # Normalize to [0,1]
        
        obj_feat = self.relu(self.obj_conv(x))
        obj_output = self.obj_head(obj_feat)
        
        return cls_output, bbox_output, obj_output

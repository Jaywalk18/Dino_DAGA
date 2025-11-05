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
    Just BN + Conv1x1, extremely simple yet effective.
    """
    def __init__(self, in_channels, num_classes, use_bn=True):
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        self.num_classes = num_classes
        
        # Official DINOv3 uses minimal architecture:
        # Just BatchNorm + 1x1 Conv + Dropout
        self.bn = nn.SyncBatchNorm(self.channels) if use_bn else nn.Identity()
        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(self.channels, num_classes, kernel_size=1)
        
        # Initialize like official implementation
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, features_list, target_size):
        # Align all features to same spatial size
        base_size = features_list[0].shape[2:]
        aligned_features = []
        for feat in features_list:
            if feat.shape[2:] != base_size:
                feat = F.interpolate(feat, size=base_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # Concatenate multi-scale features
        x = torch.cat(aligned_features, dim=1)
        
        # Simple pipeline: Dropout -> BN -> Conv
        x = self.dropout(x)
        x = self.bn(x)
        x = self.conv(x)
        
        # Upsample to target size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


class DetectionHead(nn.Module):
    """
    Detection head with deeper network.
    Outputs: class logits, bbox coordinates, objectness scores
    """
    def __init__(self, feature_dim, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(feature_dim, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Prediction heads
        self.cls_head = nn.Conv2d(256, num_anchors * num_classes, kernel_size=1)
        self.bbox_head = nn.Conv2d(256, num_anchors * 4, kernel_size=1)
        self.obj_head = nn.Conv2d(256, num_anchors, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # Initialize prediction heads
        nn.init.normal_(self.cls_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_head.bias, 0)
        
        nn.init.normal_(self.bbox_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.bbox_head.bias, 0)
        
        # Objectness head with slight positive bias
        nn.init.normal_(self.obj_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.obj_head.bias, -2.0)  # sigmoid(-2) ≈ 0.12
    
    def forward(self, x):
        # Feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        # Predictions
        cls_output = self.cls_head(x)
        bbox_output = self.bbox_head(x)
        obj_output = self.obj_head(x)
        
        return cls_output, bbox_output, obj_output

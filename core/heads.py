import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, features):
        return self.classifier(features)


class LinearSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, use_bn=True):
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        self.num_classes = num_classes
        
        if use_bn:
            self.bn = nn.BatchNorm2d(self.channels)
        else:
            self.bn = nn.Identity()
            
        self.conv = nn.Conv2d(self.channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)
        
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, features_list, target_size):
        base_size = features_list[0].shape[2:]
        aligned_features = []
        for feat in features_list:
            if feat.shape[2:] != base_size:
                feat = F.interpolate(feat, size=base_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        x = torch.cat(aligned_features, dim=1)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


class DetectionHead(nn.Module):
    def __init__(self, feature_dim, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.cls_head = nn.Conv2d(256, num_anchors * num_classes, kernel_size=1)
        self.bbox_head = nn.Conv2d(256, num_anchors * 4, kernel_size=1)
        self.obj_head = nn.Conv2d(256, num_anchors, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        
        nn.init.normal_(self.cls_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_head.bias, 0)
        nn.init.normal_(self.bbox_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.bbox_head.bias, 0)
        nn.init.normal_(self.obj_head.weight, mean=0, std=0.01)
        nn.init.constant_(self.obj_head.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        cls_output = self.cls_head(x)
        bbox_output = self.bbox_head(x)
        obj_output = self.obj_head(x)
        
        return cls_output, bbox_output, obj_output

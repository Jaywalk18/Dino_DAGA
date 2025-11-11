import torch
import torch.nn as nn


class AttentionEncoder(nn.Module):
    """
    Enhanced attention encoder with deeper architecture
    Inspired by VT-Adapter for better attention feature extraction
    """
    def __init__(self, instruction_dim=128):
        super().__init__()
        # Two-layer convolutional encoding for richer features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Two-layer FC for better instruction generation
        self.fc1 = nn.Linear(64, instruction_dim)
        self.fc2 = nn.Linear(instruction_dim, instruction_dim)
        
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm_fc = nn.LayerNorm(instruction_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, attention_map):
        """
        Args:
            attention_map: (B, H, W) normalized attention
        Returns:
            instruction: (B, instruction_dim) guidance vector
        """
        x = attention_map.unsqueeze(1)  # (B, 1, H, W)
        
        # Two-layer convolution for richer feature extraction
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        
        x = self.pool(x).flatten(1)
        
        # Two-layer FC for better instruction generation
        x = self.relu(self.fc1(x))
        x = self.norm_fc(self.fc2(x))
        
        return x


class DynamicGateGenerator(nn.Module):
    """Generate dynamic gate from instruction vector"""
    def __init__(self, instruction_dim=128, feature_dim=384):
        super().__init__()
        self.fc = nn.Linear(instruction_dim, feature_dim)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, instruction_vector):
        return torch.sigmoid(self.fc(instruction_vector))


class FeatureTransformer(nn.Module):
    """Transform features with residual connection"""
    def __init__(self, feature_dim=384, bottleneck_dim=96):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(bottleneck_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features):
        identity = features
        x = self.fc1(features)
        x = self.act(x)
        x = self.fc2(x)
        return self.norm(identity + 0.1 * x)


class DAGA(nn.Module):
    """
    Dynamic Attention-Guided Adaptation (DAGA)
    
    Core Innovation:
    - Attention-as-Guidance: Uses frozen backbone attention as spatial guidance
    - Dynamic Gating: Instance-specific adaptation via attention-guided gates
    - Gradual Adaptation: Learnable mix weight for stable training
    
    Enhanced with deeper attention encoding inspired by VT-Adapter
    """
    def __init__(self, feature_dim=384, instruction_dim=128, bottleneck_dim=96):
        super().__init__()
        self.attention_encoder = AttentionEncoder(instruction_dim)
        self.gate_generator = DynamicGateGenerator(instruction_dim, feature_dim)
        self.feature_transformer = FeatureTransformer(feature_dim, bottleneck_dim)
        self.mix_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, patch_features, attention_map):
        """
        Args:
            patch_features: (B, N, D) patch tokens
            attention_map: (B, H, W) attention guidance
        Returns:
            Adapted patch features (B, N, D)
        """
        instruction = self.attention_encoder(attention_map)
        gate = self.gate_generator(instruction).unsqueeze(1)
        
        transformed = self.feature_transformer(patch_features)
        delta = transformed - patch_features
        
        return patch_features + self.mix_weight * gate * delta

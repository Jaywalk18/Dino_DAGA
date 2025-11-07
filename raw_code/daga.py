# daga.py 
import torch
import torch.nn as nn

class AttentionEncoder(nn.Module):
    def __init__(self, instruction_dim=128):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, instruction_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(instruction_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.02)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, attention_map):
        x = attention_map.unsqueeze(1)
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        x = self.norm(self.fc(x))
        return x


class DynamicGateGenerator(nn.Module):
    def __init__(self, instruction_dim=128, feature_dim=384):
        super().__init__()
        self.fc = nn.Linear(instruction_dim, feature_dim)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, instruction_vector):
        return torch.sigmoid(self.fc(instruction_vector))


class FeatureTransformer(nn.Module):
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
    def __init__(self, feature_dim=384, instruction_dim=128, bottleneck_dim=96):
        super().__init__()
        self.attention_encoder = AttentionEncoder(instruction_dim)
        self.gate_generator = DynamicGateGenerator(instruction_dim, feature_dim)
        self.feature_transformer = FeatureTransformer(feature_dim, bottleneck_dim)
        self.mix_weight = nn.Parameter(torch.tensor(0.01))


    def forward(self, patch_features, attention_map):
        instruction = self.attention_encoder(attention_map)
        gate = self.gate_generator(instruction).unsqueeze(1)
        
        transformed = self.feature_transformer(patch_features)
        delta = transformed - patch_features
        
        return patch_features + self.mix_weight * gate * delta

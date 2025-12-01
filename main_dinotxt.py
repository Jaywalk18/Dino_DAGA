"""
DINOtxt: Text-Image Alignment Training for DINOv3 with DAGA Support
Trains a text encoder to align with frozen DINOv3 vision features
Similar to CLIP but using DINOv3 as the vision encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import argparse
from pathlib import Path
import warnings
import os
import sys
import numpy as np
from tqdm import tqdm

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment, setup_logging, finalize_experiment
from core.ddp_utils import setup_ddp, cleanup_ddp
from core.datasets import COCOCaptionsDataset

warnings.filterwarnings("ignore")


def load_clip_text_encoder_weights(text_encoder, clip_path, device='cpu'):
    """
    Load CLIP pretrained text encoder weights into SimpleTextEncoder.
    
    CLIP ViT-B/16 text encoder structure:
    - token_embedding.weight: (49408, 512)
    - positional_embedding: (77, 512)
    - transformer.resblocks.{0-11}.attn.in_proj_{weight,bias}
    - transformer.resblocks.{0-11}.attn.out_proj.{weight,bias}
    - transformer.resblocks.{0-11}.ln_1.{weight,bias}
    - transformer.resblocks.{0-11}.ln_2.{weight,bias}
    - transformer.resblocks.{0-11}.mlp.c_fc.{weight,bias}
    - transformer.resblocks.{0-11}.mlp.c_proj.{weight,bias}
    - ln_final.{weight,bias}
    - text_projection: (512, 512)
    
    Our SimpleTextEncoder uses nn.TransformerEncoderLayer which has:
    - self_attn.in_proj_{weight,bias}
    - self_attn.out_proj.{weight,bias}
    - norm1.{weight,bias} (corresponds to ln_1)
    - norm2.{weight,bias} (corresponds to ln_2)
    - linear1.{weight,bias} (corresponds to mlp.c_fc)
    - linear2.{weight,bias} (corresponds to mlp.c_proj)
    """
    print(f"Loading CLIP text encoder weights from {clip_path}...")
    
    # Load CLIP model (it's a TorchScript archive)
    clip_model = torch.jit.load(clip_path, map_location=device)
    clip_state = clip_model.state_dict()
    
    # Build mapping from CLIP to our model
    new_state_dict = {}
    
    # Token embedding
    new_state_dict['token_embedding.weight'] = clip_state['token_embedding.weight']
    
    # Positional embedding
    new_state_dict['position_embedding'] = clip_state['positional_embedding']
    
    # Final layer norm
    new_state_dict['ln_final.weight'] = clip_state['ln_final.weight']
    new_state_dict['ln_final.bias'] = clip_state['ln_final.bias']
    
    # Text projection
    # CLIP: text_projection is (512, 512), our model has Linear(512, 512, bias=False)
    new_state_dict['text_projection.weight'] = clip_state['text_projection'].T  # Transpose for nn.Linear
    
    # Transformer layers
    num_layers = text_encoder.transformer.num_layers
    for i in range(num_layers):
        clip_prefix = f'transformer.resblocks.{i}'
        our_prefix = f'transformer.layers.{i}'
        
        # Attention layers
        new_state_dict[f'{our_prefix}.self_attn.in_proj_weight'] = clip_state[f'{clip_prefix}.attn.in_proj_weight']
        new_state_dict[f'{our_prefix}.self_attn.in_proj_bias'] = clip_state[f'{clip_prefix}.attn.in_proj_bias']
        new_state_dict[f'{our_prefix}.self_attn.out_proj.weight'] = clip_state[f'{clip_prefix}.attn.out_proj.weight']
        new_state_dict[f'{our_prefix}.self_attn.out_proj.bias'] = clip_state[f'{clip_prefix}.attn.out_proj.bias']
        
        # Layer norms (CLIP uses ln_1/ln_2, our TransformerEncoderLayer uses norm1/norm2)
        new_state_dict[f'{our_prefix}.norm1.weight'] = clip_state[f'{clip_prefix}.ln_1.weight']
        new_state_dict[f'{our_prefix}.norm1.bias'] = clip_state[f'{clip_prefix}.ln_1.bias']
        new_state_dict[f'{our_prefix}.norm2.weight'] = clip_state[f'{clip_prefix}.ln_2.weight']
        new_state_dict[f'{our_prefix}.norm2.bias'] = clip_state[f'{clip_prefix}.ln_2.bias']
        
        # MLP layers (CLIP uses c_fc/c_proj, our TransformerEncoderLayer uses linear1/linear2)
        new_state_dict[f'{our_prefix}.linear1.weight'] = clip_state[f'{clip_prefix}.mlp.c_fc.weight']
        new_state_dict[f'{our_prefix}.linear1.bias'] = clip_state[f'{clip_prefix}.mlp.c_fc.bias']
        new_state_dict[f'{our_prefix}.linear2.weight'] = clip_state[f'{clip_prefix}.mlp.c_proj.weight']
        new_state_dict[f'{our_prefix}.linear2.bias'] = clip_state[f'{clip_prefix}.mlp.c_proj.bias']
    
    # Load weights with strict=False to handle any missing keys
    missing, unexpected = text_encoder.load_state_dict(new_state_dict, strict=False)
    
    print(f"  Loaded {len(new_state_dict)} weight tensors from CLIP")
    if missing:
        print(f"  Missing keys (will use random init): {missing}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    
    return text_encoder
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


class SimpleTextEncoder(nn.Module):
    """
    Transformer-based text encoder (following official DINOtxt config)
    
    Key features:
    - Uses argmax pooling: takes features at EOS token position
    - EOS token has the highest token ID in CLIP tokenizer (49407)
    """
    def __init__(self, vocab_size=49408, embed_dim=512, num_layers=12, num_heads=8, max_seq_len=77):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Transformer layers (using causal attention like official config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Linear projection (official: text_model_use_linear_projection=true)
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, text_tokens):
        """
        Args:
            text_tokens: (B, seq_len) - token indices
        Returns:
            text_features: (B, embed_dim) - normalized text features
        """
        B, seq_len = text_tokens.shape
        
        # Embed tokens
        x = self.token_embedding(text_tokens)  # (B, seq_len, embed_dim)
        
        # Add positional embeddings
        x = x + self.position_embedding[:seq_len].unsqueeze(0)
        
        # Create causal mask for autoregressive modeling
        # This ensures each position only attends to earlier positions
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
            diagonal=1
        )
        
        # Transformer encoding with causal mask
        x = self.transformer(x, mask=causal_mask)
        
        # Argmax pooling: find position of highest token ID (EOS token)
        # In CLIP tokenizer, EOS token (49407) is at the end of actual text
        eos_positions = text_tokens.argmax(dim=-1)  # (B,) - position of max token ID
        
        # Gather features at EOS positions
        # x: (B, seq_len, embed_dim), eos_positions: (B,)
        batch_indices = torch.arange(B, device=x.device)
        x = x[batch_indices, eos_positions]  # (B, embed_dim)
        
        # Final layer norm and projection
        x = self.ln_final(x)
        x = self.text_projection(x)
        
        # L2 normalize
        x = F.normalize(x, dim=-1)
        
        return x


class VisionHeadBlock(nn.Module):
    """
    Trainable head block (similar to ViT block but simpler)
    Used on top of frozen backbone features for fine-tuning.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.drop_path1 = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )
        self.drop_path2 = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn_out)
        
        # MLP with residual
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionEncoder(nn.Module):
    """
    Frozen DINOv3 vision encoder with trainable head blocks (official DINOtxt architecture)
    
    Architecture (following official DINOtxt config):
    1. Frozen DINOv3 backbone extracts patch tokens
    2. Trainable head blocks (num_head_blocks=2) process patch tokens
    3. Mean pooling over patch tokens
    4. Linear projection to output dimension
    """
    def __init__(self, vit_model, use_daga=False, daga_layers=None, output_dim=512,
                 num_head_blocks=2, head_drop_path=0.3, use_patch_tokens=True):
        super().__init__()
        self.vit_model = vit_model
        self.use_daga = use_daga
        self.daga_layers = daga_layers if daga_layers else []
        self.feature_dim = vit_model.embed_dim
        self.output_dim = output_dim
        self.use_patch_tokens = use_patch_tokens
        
        # Freeze all ViT parameters
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        # Trainable head blocks (official: num_head_blocks=2, drop_path=0.3)
        self.head_blocks = nn.ModuleList([
            VisionHeadBlock(
                dim=self.feature_dim,
                num_heads=8,
                mlp_ratio=4.0,
                drop_path=head_drop_path
            ) for _ in range(num_head_blocks)
        ])
        
        # Final projection (official: use_linear_projection=False means no extra projection)
        # But we need projection to match text dimension
        self.vision_proj = nn.Linear(self.feature_dim, output_dim, bias=False)
        nn.init.normal_(self.vision_proj.weight, std=self.feature_dim ** -0.5)
        
        # Final layer norm
        self.ln_post = nn.LayerNorm(self.feature_dim)
        
        # Initialize DAGA modules if enabled
        if self.use_daga and self.daga_layers:
            from core.daga import DAGA
            
            self.daga_modules = nn.ModuleDict({
                str(i): DAGA(feature_dim=self.feature_dim) for i in self.daga_layers
            })
            for param in self.daga_modules.parameters():
                param.requires_grad = True
            
            self.daga_guidance_layer_idx = len(vit_model.blocks) - 1
            print(f"  DAGA initialized: layers {self.daga_layers}, guidance from layer {self.daga_guidance_layer_idx}")
        
        print(f"  Vision encoder: {num_head_blocks} head blocks, drop_path={head_drop_path}")
    
    def forward(self, images, return_attention=False):
        """
        Args:
            images: (B, 3, H, W)
            return_attention: if True, return attention maps for visualization
        Returns:
            vision_features: (B, output_dim) - normalized vision features
            attention_info: dict with attention maps (only if return_attention=True)
        """
        attention_info = None
        
        # Get features from frozen backbone
        with torch.no_grad():
            features = self.vit_model.forward_features(images)
            if self.use_patch_tokens:
                x = features["x_norm_patchtokens"]  # (B, N, D) - use patch tokens
            else:
                x = features["x_norm_clstoken"].unsqueeze(1)  # (B, 1, D) - use CLS token
        
        if self.use_daga and self.daga_layers:
            # Get additional info for DAGA
            with torch.no_grad():
                tokens, (H, W) = self.vit_model.prepare_tokens_with_masks(images)
            
            from core.backbones import compute_daga_guidance_map, get_attention_map
            daga_guidance_map = compute_daga_guidance_map(
                self.vit_model, tokens, H, W, self.daga_guidance_layer_idx
            )
            
            # Collect attention maps for visualization
            if return_attention:
                attention_info = {
                    'H': H,
                    'W': W,
                    'daga_guidance_map': daga_guidance_map.detach(),
                    'baseline_attn': None,
                    'adapted_attn': None,
                }
                # Get baseline attention from frozen backbone (before DAGA)
                with torch.no_grad():
                    x_proc = tokens.clone()
                    for i in range(self.daga_guidance_layer_idx + 1):
                        rope_sincos = (
                            self.vit_model.rope_embed(H=H, W=W)
                            if self.vit_model.rope_embed
                            else None
                        )
                        if i == self.daga_guidance_layer_idx:
                            attention_info['baseline_attn'] = get_attention_map(
                                self.vit_model.blocks[i], x_proc
                            ).detach()
                        x_proc = self.vit_model.blocks[i](x_proc, rope_sincos)
            
            # Apply DAGA to features
            for layer_idx in self.daga_layers:
                x = self.daga_modules[str(layer_idx)](x, daga_guidance_map)
            
            # For visualization: compute "adapted attention" from head block
            if return_attention and len(self.head_blocks) > 0:
                x_for_attn = x.clone()
                x_norm = self.head_blocks[0].norm1(x_for_attn)
                _, attn_weights = self.head_blocks[0].attn(
                    x_norm, x_norm, x_norm, 
                    need_weights=True, average_attn_weights=True
                )
                # attn_weights: (B, N, N) - attention over patch tokens
                adapted_attn = attn_weights.mean(dim=1)  # (B, N)
                B, N = adapted_attn.shape
                adapted_attn = adapted_attn.view(B, H, W)
                attention_info['adapted_attn'] = adapted_attn.detach()
        
        # Apply trainable head blocks
        for block in self.head_blocks:
            x = block(x)
        
        # Mean pooling (official: patch_tokens_pooler_type=mean)
        x = x.mean(dim=1)  # (B, D)
        
        # Final layer norm and projection
        x = self.ln_post(x)
        x = self.vision_proj(x)
        
        # L2 normalize
        x = F.normalize(x, dim=-1)
        
        if return_attention:
            return x, attention_info
        return x


class DINOtxtModel(nn.Module):
    """DINOtxt: Vision-Language alignment model"""
    def __init__(self, vision_encoder, text_encoder, vision_dim, text_dim):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Learnable temperature parameter (initialized to ln(1/0.07) ≈ 2.66)
        # Following CLIP: clamp to prevent extreme values
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_max = np.log(100)  # Max temperature = 100
        self.logit_scale_min = np.log(1)    # Min temperature = 1
        
        # Projection layers if dimensions don't match
        if vision_dim != text_dim:
            self.vision_proj = nn.Linear(vision_dim, text_dim)
            self.text_proj = nn.Linear(text_dim, text_dim)
        else:
            self.vision_proj = nn.Identity()
            self.text_proj = nn.Identity()
    
    def forward(self, images, text_tokens):
        """
        Args:
            images: (B, 3, H, W)
            text_tokens: (B, seq_len)
        Returns:
            vision_features: (B, D)
            text_features: (B, D)
            logit_scale: scalar (clamped)
        """
        # Encode vision and text
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(text_tokens)
        
        # Project to common space
        vision_features = self.vision_proj(vision_features)
        text_features = self.text_proj(text_features)
        
        # Normalize (encoders already normalize, but ensure consistency)
        vision_features = F.normalize(vision_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Clamp logit_scale to prevent collapse (CRITICAL for stable training)
        logit_scale = torch.clamp(self.logit_scale, self.logit_scale_min, self.logit_scale_max).exp()
        
        return vision_features, text_features, logit_scale


def clip_loss(vision_features, text_features, logit_scale):
    """
    Contrastive loss for vision-language alignment (CLIP-style)
    Args:
        vision_features: (B, D) - normalized
        text_features: (B, D) - normalized
        logit_scale: scalar temperature
    """
    B = vision_features.shape[0]
    
    # Compute similarity matrix
    logits_per_image = logit_scale * vision_features @ text_features.t()  # (B, B)
    logits_per_text = logits_per_image.t()  # (B, B)
    
    # Labels: diagonal should be 1 (matching pairs)
    labels = torch.arange(B, device=vision_features.device)
    
    # Cross-entropy loss
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    loss = (loss_i + loss_t) / 2
    
    return loss


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="DINOtxt Training with DAGA")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16", help="DINOv3 model architecture")
    parser.add_argument("--pretrained_path", type=str, default="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", help="Path to pretrained checkpoint")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="coco_captions", help="Dataset name")
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--sample_ratio", type=float, default=None, help="Ratio of training data to use")
    
    # DAGA arguments
    parser.add_argument("--use_daga", action="store_true", help="Use DAGA")
    parser.add_argument("--daga_layers", type=int, nargs="+", default=[11], help="Layers to apply DAGA")
    
    # Vision encoder arguments (official DINOtxt config)
    parser.add_argument("--num_head_blocks", type=int, default=2, help="Number of trainable head blocks (official: 2)")
    parser.add_argument("--head_drop_path", type=float, default=0.3, help="Drop path rate for head blocks (official: 0.3)")
    parser.add_argument("--use_patch_tokens", action="store_true", default=True, help="Use patch tokens instead of CLS (official: True)")
    
    # Text encoder arguments
    parser.add_argument("--text_embed_dim", type=int, default=512, help="Text embedding dimension")
    parser.add_argument("--text_num_layers", type=int, default=12, help="Number of text transformer layers")
    parser.add_argument("--text_num_heads", type=int, default=8, help="Number of text attention heads")
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=77, help="Maximum sequence length")
    
    # Loss arguments
    parser.add_argument("--clip_loss_weight", type=float, default=1.0, help="CLIP loss weight")
    
    # CLIP pretrained text encoder
    parser.add_argument("--clip_pretrained_path", type=str, default=None, 
                        help="Path to CLIP pretrained weights (e.g., checkpoints/clip_vit_b16.pt)")
    
    # Training arguments
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    
    # Output arguments
    parser.add_argument("--output_dir", default="./outputs/dinotxt", help="Output directory")
    parser.add_argument("--log_freq", type=int, default=1, help="Evaluation frequency (default: every epoch)")
    parser.add_argument("--enable_visualization", action="store_true", help="Enable retrieval visualization at log_freq epochs")
    parser.add_argument("--num_vis_samples", type=int, default=3, help="Number of samples to visualize (default: 3)")
    
    # SwanLab logging arguments
    parser.add_argument("--swanlab_name", type=str, default=None, help="SwanLab experiment name")
    parser.add_argument("--swanlab_mode", type=str, default="cloud", help="SwanLab mode (cloud/local/disabled)")
    
    return parser.parse_args()


class DummyImageTextDataset(torch.utils.data.Dataset):
    """
    Dummy image-text paired dataset for testing without real data
    """
    def __init__(self, num_samples=1000, input_size=224, max_seq_len=77):
        self.num_samples = num_samples
        self.input_size = input_size
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = torch.randn(3, self.input_size, self.input_size)
        text_tokens = torch.randint(0, 30000, (self.max_seq_len,))
        return image, text_tokens


def train_one_epoch(model, train_loader, optimizer, scheduler, device, args, epoch, rank):
    """Train for one epoch with learning rate scheduling"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    is_main_process = (rank == 0)
    
    if is_main_process:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, (images, text_tokens) in enumerate(pbar):
        images = images.to(device)
        text_tokens = text_tokens.to(device)
        
        # Forward pass
        vision_features, text_features, logit_scale = model(images, text_tokens)
        
        # Compute CLIP loss
        loss = clip_loss(vision_features, text_features, logit_scale)
        loss = loss * args.clip_loss_weight
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Step scheduler per iteration
        
        total_loss += loss.item()
        num_batches += 1
        
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': loss.item(),
                'logit_scale': logit_scale.item(),
                'lr': f'{current_lr:.2e}'
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def visualize_retrieval(images, captions, vision_features, text_features, 
                        output_dir, epoch, num_samples=3):
    """
    Visualize image-text retrieval results
    
    Args:
        images: list of image tensors (3, H, W)
        captions: list of caption strings
        vision_features: (N, D) vision feature matrix
        text_features: (N, D) text feature matrix
        output_dir: directory to save visualizations
        epoch: current epoch
        num_samples: number of samples to visualize
    
    Returns:
        list of matplotlib figures for SwanLab logging
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    vis_save_path = Path(output_dir) / "visualizations"
    vis_save_path.mkdir(parents=True, exist_ok=True)
    
    vis_figs = []
    num_samples = min(num_samples, len(images))
    
    # Compute similarity matrix
    similarity = vision_features @ text_features.t()
    
    for idx in range(num_samples):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Retrieval Results - Epoch {epoch} - Sample {idx+1}", 
                    fontsize=14, fontweight="bold")
        
        # Denormalize image
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = np.clip(std * img + mean, 0, 1)
        
        # Left: Image with its ground truth caption
        axes[0].imshow(img)
        gt_caption = captions[idx] if idx < len(captions) else f"Caption {idx}"
        axes[0].set_title(f"Query Image\nGT: {gt_caption[:50]}...", fontsize=10)
        axes[0].axis("off")
        
        # Right: Top-5 retrieved captions for this image
        img_sim = similarity[idx]  # similarity of this image to all texts
        top5_indices = img_sim.topk(5).indices.tolist()
        
        retrieved_text = "Top-5 Retrieved Captions:\n\n"
        for rank, t_idx in enumerate(top5_indices):
            sim_score = img_sim[t_idx].item()
            cap = captions[t_idx] if t_idx < len(captions) else f"Caption {t_idx}"
            is_correct = "✓" if t_idx == idx else ""
            retrieved_text += f"{rank+1}. [{sim_score:.3f}] {cap[:60]}... {is_correct}\n\n"
        
        axes[1].text(0.05, 0.95, retrieved_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].set_title("Image → Text Retrieval", fontsize=12)
        axes[1].axis("off")
        
        plt.tight_layout()
        
        save_path = vis_save_path / f"epoch_{epoch}_sample_{idx+1}_retrieval.png"
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        vis_figs.append(fig)
    
    print(f"  ✓ Saved {num_samples} retrieval visualizations")
    return vis_figs


def visualize_daga_attention(model, images, output_dir, epoch, num_samples=3):
    """
    Visualize DAGA attention maps for DINOtxt.
    
    Shows 3 panels:
    1. Frozen Backbone Attention
    2. Adapted Attention (after DAGA)
    3. Adapted Attention Overlay on Image
    
    Args:
        model: DINOtxt model (with DAGA enabled)
        images: (B, 3, H, W) batch of images
        output_dir: directory to save visualizations
        epoch: current epoch
        num_samples: number of samples to visualize
    
    Returns:
        list of matplotlib figures for SwanLab logging
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    from core.backbones import process_attention_weights
    
    # Get actual model from DDP wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    vision_encoder = actual_model.vision_encoder
    
    # Check if DAGA is enabled
    if not getattr(vision_encoder, 'use_daga', False):
        print("  ⚠ DAGA not enabled, skipping attention visualization")
        return []
    
    vis_figs = []
    vis_save_path = Path(output_dir) / "visualizations"
    vis_save_path.mkdir(parents=True, exist_ok=True)
    
    num_samples = min(num_samples, images.shape[0])
    sample_images = images[:num_samples]
    
    with torch.no_grad():
        # Get attention info from vision encoder
        _, attention_info = vision_encoder(sample_images, return_attention=True)
        
        if attention_info is None:
            print("  ⚠ Could not get attention maps")
            return []
        
        H, W = attention_info['H'], attention_info['W']
        num_patches = H * W
        baseline_attn = attention_info.get('baseline_attn')
        adapted_attn = attention_info.get('adapted_attn')  # (B, H, W)
        
        # Process baseline attention
        baseline_attn_np = None
        if baseline_attn is not None:
            baseline_attn_np = process_attention_weights(baseline_attn, num_patches, H, W)
        
        # Process adapted attention
        adapted_attn_np = None
        if adapted_attn is not None:
            adapted_attn_np = adapted_attn.cpu().numpy()
            # Normalize for visualization
            for i in range(adapted_attn_np.shape[0]):
                min_val = adapted_attn_np[i].min()
                max_val = adapted_attn_np[i].max()
                adapted_attn_np[i] = (adapted_attn_np[i] - min_val) / (max_val - min_val + 1e-8)
        
        # Denormalize images
        images_np = sample_images.cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        for idx in range(num_samples):
            # Create figure with 1x3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"DAGA Attention Analysis - Epoch {epoch} - Sample {idx+1}", 
                        fontsize=14, fontweight="bold")
            
            # Denormalize image
            img = images_np[idx].transpose(1, 2, 0)
            img = np.clip(std * img + mean, 0, 1)
            
            # Panel 1: Frozen Backbone Attention
            if baseline_attn_np is not None:
                im1 = axes[0].imshow(baseline_attn_np[idx], cmap="viridis", vmin=0, vmax=1)
                axes[0].set_title(f"Frozen Backbone Attn\n(L{vision_encoder.daga_guidance_layer_idx})", fontsize=12)
                plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            else:
                axes[0].imshow(img)
                axes[0].set_title("Original Image", fontsize=12)
            axes[0].axis("off")
            
            # Panel 2: Adapted Attention (after DAGA)
            if adapted_attn_np is not None:
                im2 = axes[1].imshow(adapted_attn_np[idx], cmap="viridis", vmin=0, vmax=1)
                axes[1].set_title("Adapted Attn\n(After DAGA)", fontsize=12)
                plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            else:
                axes[1].text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=14)
                axes[1].set_title("Adapted Attn", fontsize=12)
            axes[1].axis("off")
            
            # Panel 3: Adapted Attention Overlay on Image (use adapted_attn, not guidance)
            axes[2].imshow(img)
            if adapted_attn_np is not None:
                attn_map = adapted_attn_np[idx]
                if attn_map.shape != img.shape[:2]:
                    zoom_h = img.shape[0] / attn_map.shape[0]
                    zoom_w = img.shape[1] / attn_map.shape[1]
                    attn_resized = zoom(attn_map, (zoom_h, zoom_w), order=1)
                else:
                    attn_resized = attn_map
                im3 = axes[2].imshow(attn_resized, cmap="jet", alpha=0.5, vmin=0, vmax=1)
                plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            axes[2].set_title("Adapted Attn Overlay", fontsize=12)
            axes[2].axis("off")
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            save_path = vis_save_path / f"epoch_{epoch}_sample_{idx+1}_daga_attention.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            vis_figs.append(fig)
        
        # Print DAGA mix weights
        print(f"  DAGA Mix Weights at Epoch {epoch}:")
        for layer_idx, daga_module in vision_encoder.daga_modules.items():
            weight_val = daga_module.mix_weight.item()
            print(f"    Layer {layer_idx}: {weight_val:.6f}")
    
    print(f"  ✓ Saved {num_samples} DAGA attention visualizations")
    return vis_figs


def evaluate(model, test_loader, device, rank, collect_for_vis=False, num_vis_samples=3, dataset=None):
    """
    Evaluate with proper gathering of all features across GPUs.
    Computes global R@1 by gathering all features before computing metrics.
    """
    model.eval()
    
    all_vision_features = []
    all_text_features = []
    vis_images = []
    all_captions = []
    
    is_main_process = (rank == 0)
    world_size = dist.get_world_size()
    
    with torch.no_grad():
        if is_main_process:
            pbar = tqdm(test_loader, desc="Evaluating")
        else:
            pbar = test_loader
        
        for batch_idx, (images, text_tokens) in enumerate(pbar):
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            
            vision_features, text_features, _ = model(images, text_tokens)
            
            all_vision_features.append(vision_features)
            all_text_features.append(text_features)
            
            if collect_for_vis and is_main_process and batch_idx == 0:
                for i in range(min(num_vis_samples, images.shape[0])):
                    vis_images.append(images[i].cpu())
                    if dataset is not None and hasattr(dataset, 'get_caption'):
                        try:
                            all_captions.append(dataset.get_caption(i))
                        except:
                            all_captions.append(f"Sample {i+1}")
    
    # Concatenate local features
    vision_local = torch.cat(all_vision_features, dim=0)
    text_local = torch.cat(all_text_features, dim=0)
    
    # Gather all features from all GPUs
    local_size = torch.tensor([vision_local.shape[0]], device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)
    
    # Pad to max size for gathering
    if vision_local.shape[0] < max_size:
        pad_size = max_size - vision_local.shape[0]
        vision_local = torch.cat([vision_local, torch.zeros(pad_size, vision_local.shape[1], device=device)], dim=0)
        text_local = torch.cat([text_local, torch.zeros(pad_size, text_local.shape[1], device=device)], dim=0)
    
    # Gather from all GPUs
    vision_list = [torch.zeros_like(vision_local) for _ in range(world_size)]
    text_list = [torch.zeros_like(text_local) for _ in range(world_size)]
    dist.all_gather(vision_list, vision_local)
    dist.all_gather(text_list, text_local)
    
    # Interleave to restore original order (DistributedSampler assigns round-robin indices)
    vision_all = []
    text_all = []
    total_samples = sum(s.item() for s in all_sizes)
    for i in range(max_size):
        for r in range(world_size):
            if i < all_sizes[r].item():
                vision_all.append(vision_list[r][i])
                text_all.append(text_list[r][i])
    
    vision_features = torch.stack(vision_all, dim=0).cpu()
    text_features = torch.stack(text_all, dim=0).cpu()
    
    # Compute global R@1 metrics
    similarity = vision_features @ text_features.t()
    
    # I2T: for each image, find the most similar text
    _, indices = similarity.topk(1, dim=1)
    correct = (indices.squeeze() == torch.arange(len(vision_features))).sum().item()
    i2t_acc = correct / len(vision_features)
    
    # T2I: for each text, find the most similar image
    _, indices = similarity.t().topk(1, dim=1)
    correct = (indices.squeeze() == torch.arange(len(text_features))).sum().item()
    t2i_acc = correct / len(text_features)
    
    if is_main_process:
        print(f"  Evaluated on {len(vision_features)} total samples (gathered from {world_size} GPUs)")
    
    if collect_for_vis:
        return i2t_acc, t2i_acc, vis_images, all_captions, vision_features, text_features
    return i2t_acc, t2i_acc


def main():
    # Setup DDP
    local_rank, rank, world_size = setup_ddp()
    is_main_process = (rank == 0)
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Parse arguments
    args = parse_arguments()
    setup_environment(args.seed + rank)
    
    if is_main_process:
        experiment_name = setup_logging(args, task_name="dinotxt")
        output_dir = Path(args.output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"DINOtxt Training with {world_size} GPUs")
        print(f"DAGA: {'Enabled' if args.use_daga else 'Disabled'}")
        if args.use_daga:
            print(f"DAGA Layers: {args.daga_layers}")
        print(f"Text Encoder: {args.text_num_layers} layers, {args.text_embed_dim} dim")
        print(f"{'='*70}\n")
    
    # Broadcast output_dir to all processes
    output_dir_list = [str(output_dir)] if is_main_process else [None]
    dist.barrier()
    dist.broadcast_object_list(output_dir_list, src=0)
    if not is_main_process:
        output_dir = Path(output_dir_list[0])
    
    # Load vision model
    if is_main_process:
        print(f"Loading DINOv3 vision encoder '{args.model_name}'...")
    
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    # Create datasets first to get correct vocab_size
    if is_main_process:
        print(f"Loading datasets...")
    
    if args.data_path and os.path.exists(args.data_path):
        # Use real COCO Captions dataset
        try:
            train_dataset = COCOCaptionsDataset(
                data_path=args.data_path,
                split='train',
                input_size=args.input_size,
                max_seq_len=args.max_seq_len,
                vocab_size=args.vocab_size,
                sample_ratio=args.sample_ratio,
            )
            test_dataset = COCOCaptionsDataset(
                data_path=args.data_path,
                split='val',
                input_size=args.input_size,
                max_seq_len=args.max_seq_len,
                vocab_size=args.vocab_size,
            )
            # Get actual vocab_size from dataset (may be 49408 for CLIP tokenizer)
            actual_vocab_size = getattr(train_dataset, 'vocab_size', args.vocab_size)
            if is_main_process:
                print(f"✓ Loaded COCO Captions: {len(train_dataset)} train, {len(test_dataset)} val")
                print(f"  Vocab size: {actual_vocab_size}")
        except Exception as e:
            if is_main_process:
                print(f"⚠️ Failed to load COCO Captions: {e}")
                print(f"  Falling back to dummy dataset...")
            train_dataset = DummyImageTextDataset(10000, args.input_size, args.max_seq_len)
            test_dataset = DummyImageTextDataset(1000, args.input_size, args.max_seq_len)
            actual_vocab_size = args.vocab_size
    else:
        # Use dummy dataset for testing
        if is_main_process:
            print(f"⚠️ No data_path provided or path doesn't exist, using dummy dataset")
        train_dataset = DummyImageTextDataset(10000, args.input_size, args.max_seq_len)
        test_dataset = DummyImageTextDataset(1000, args.input_size, args.max_seq_len)
        actual_vocab_size = args.vocab_size
    
    # Create text encoder with correct vocab_size
    if is_main_process:
        print(f"Creating text encoder (vocab_size={actual_vocab_size})...")
    
    text_encoder = SimpleTextEncoder(
        vocab_size=actual_vocab_size,
        embed_dim=args.text_embed_dim,
        num_layers=args.text_num_layers,
        num_heads=args.text_num_heads,
        max_seq_len=args.max_seq_len,
    )
    
    # Load CLIP pretrained weights if provided
    if args.clip_pretrained_path and os.path.exists(args.clip_pretrained_path):
        if is_main_process:
            print(f"\n{'='*50}")
            print(f"Loading CLIP pretrained text encoder...")
            print(f"{'='*50}")
        text_encoder = load_clip_text_encoder_weights(
            text_encoder, 
            args.clip_pretrained_path,
            device='cpu'  # Load on CPU first
        )
        if is_main_process:
            print(f"✓ CLIP text encoder weights loaded successfully!")
            print(f"{'='*50}\n")
    elif args.clip_pretrained_path:
        if is_main_process:
            print(f"⚠️ CLIP pretrained path not found: {args.clip_pretrained_path}")
            print(f"   Training text encoder from scratch...")
    
    # Create VisionEncoder with trainable head blocks (official DINOtxt architecture)
    vision_encoder = VisionEncoder(
        vit_model, 
        use_daga=args.use_daga, 
        daga_layers=args.daga_layers,
        output_dim=args.text_embed_dim,  # Match text encoder dimension
        num_head_blocks=args.num_head_blocks,  # Official: 2
        head_drop_path=args.head_drop_path,    # Official: 0.3
        use_patch_tokens=args.use_patch_tokens  # Official: True
    )
    
    # Create DINOtxt model
    model = DINOtxtModel(
        vision_encoder,
        text_encoder,
        vision_dim=args.text_embed_dim,  # VisionEncoder now outputs text_embed_dim
        text_dim=args.text_embed_dim,
    )
    model.to(device)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process:
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Wrap with DDP
    # Note: find_unused_parameters=False is sufficient since DAGA is always used in forward pass when enabled
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # DAGA parameters are always used when enabled
    )
    
    if is_main_process:
        print(f"✓ Models loaded and wrapped with DDP\n")
    
    # Create dataloaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Setup optimizer (only optimize text encoder and projections)
    # Note: Don't scale LR by world_size for contrastive learning - it can destabilize training
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,  # Removed world_size scaling for stability
        weight_decay=args.weight_decay
    )
    
    # Setup learning rate scheduler with warmup + cosine decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    if is_main_process:
        print(f"LR Schedule: warmup {args.warmup_epochs} epochs, then cosine decay")
        print(f"Starting training...\n")
    
    best_i2t_acc = 0.0
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        # Train
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, args, epoch, rank)
        
        # Log training metrics every epoch to SwanLab
        if is_main_process:
            try:
                import swanlab
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                # Get logit_scale from model
                logit_scale = model.module.logit_scale.exp().item()
                
                swanlab.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "learning_rate": current_lr,
                    "logit_scale": logit_scale,
                }, step=epoch + 1)
            except Exception as e:
                pass  # Silent fail for per-epoch logging
        
        # Evaluate R@1 every epoch (all processes participate)
        should_visualize = args.enable_visualization and is_main_process and ((epoch + 1) % args.log_freq == 0)
            vis_images, vis_captions, vis_vision_feat, vis_text_feat = [], [], None, None
            
            # All processes participate in evaluation using DDP test_loader
            if should_visualize:
                result = evaluate(
                    model, test_loader, device, rank,
                    collect_for_vis=True, 
                    num_vis_samples=args.num_vis_samples,
                    dataset=test_dataset
                )
                i2t_acc, t2i_acc, vis_images, vis_captions, vis_vision_feat, vis_text_feat = result
            else:
                i2t_acc, t2i_acc = evaluate(model, test_loader, device, rank)
            
            if is_main_process:
                print(f"\nEpoch {epoch+1}/{args.epochs}:")
                print(f"  Train Loss: {avg_loss:.4f}")
                print(f"  Image-to-Text R@1: {i2t_acc*100:.2f}%")
                print(f"  Text-to-Image R@1: {t2i_acc*100:.2f}%")
                
            # Log R@1 metrics to SwanLab every epoch
            try:
                import swanlab
                swanlab.log({
                    "i2t_r1": i2t_acc * 100,
                    "t2i_r1": t2i_acc * 100,
                }, step=epoch + 1)
            except Exception as e:
                pass  # Silent fail
            
            # Generate visualizations only at log_freq intervals
            if should_visualize:
                vis_figs = []
                
                # 1. Retrieval visualizations
                if vis_images:
                    print("📊 Generating retrieval visualizations...")
                    vis_figs = visualize_retrieval(
                        vis_images, vis_captions, 
                        vis_vision_feat, vis_text_feat,
                        output_dir, epoch + 1, 
                        num_samples=args.num_vis_samples
                    )
                
                # 2. DAGA attention map visualizations (if DAGA is enabled)
                actual_model = model.module if hasattr(model, 'module') else model
                if getattr(actual_model.vision_encoder, 'use_daga', False) and vis_images:
                    print("📊 Generating DAGA attention visualizations...")
                    # Stack vis_images to tensor
                    vis_images_tensor = torch.stack(vis_images).to(device)
                    daga_figs = visualize_daga_attention(
                        model, vis_images_tensor,
                        output_dir, epoch + 1,
                        num_samples=args.num_vis_samples
                    )
                    vis_figs.extend(daga_figs)
                
                # Log visualizations to SwanLab
                if vis_figs:
                try:
                    import swanlab
                        import matplotlib.pyplot as plt
                        swanlab.log({
                            "visualizations": [swanlab.Image(fig) for fig in vis_figs]
                        }, step=epoch + 1)
                        # Close figures to free memory
                        for fig in vis_figs:
                            plt.close(fig)
                except Exception as e:
                        print(f"  Warning: SwanLab visualization logging failed: {e}")
                
                # Save best model
                if i2t_acc > best_i2t_acc:
                    best_i2t_acc = i2t_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'i2t_acc': i2t_acc,
                        't2i_acc': t2i_acc,
                    }, output_dir / "best_model.pth")
                    print(f"  ✓ Saved best model (I2T R@1: {best_i2t_acc*100:.2f}%)")
        
        dist.barrier()
    
    if is_main_process:
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Best Image-to-Text R@1: {best_i2t_acc*100:.2f}%")
        print(f"{'='*70}\n")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()


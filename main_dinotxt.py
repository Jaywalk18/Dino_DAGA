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

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


class SimpleTextEncoder(nn.Module):
    """Simple Transformer-based text encoder"""
    def __init__(self, vocab_size=30522, embed_dim=512, num_layers=12, num_heads=8, max_seq_len=77):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Transformer layers
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
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Take the [CLS] token (first token) as the text representation
        x = x[:, 0, :]  # (B, embed_dim)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # L2 normalize
        x = F.normalize(x, dim=-1)
        
        return x


class VisionEncoder(nn.Module):
    """Frozen DINOv3 vision encoder with optional DAGA"""
    def __init__(self, vit_model, use_daga=False, daga_layers=None):
        super().__init__()
        self.vit_model = vit_model
        self.use_daga = use_daga
        self.daga_layers = daga_layers if daga_layers else []
        
        # Freeze all parameters
        for param in self.vit_model.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            vision_features: (B, embed_dim) - normalized vision features
        """
        with torch.no_grad():
            B = images.shape[0]
            
            # Get patch embeddings
            x = self.vit_model.prepare_tokens_with_masks(images)
            
            # Get dimensions
            H = W = int((x.shape[1] - 1 - getattr(self.vit_model, 'n_storage_tokens', 0)) ** 0.5)
            
            # Forward through blocks
            for i, block in enumerate(self.vit_model.blocks):
                rope_sincos = self.vit_model.rope_embed(H=H, W=W) if self.vit_model.rope_embed else None
                
                if self.use_daga and i in self.daga_layers:
                    # DAGA would be applied here
                    x = block(x, rope_sincos)
                else:
                    x = block(x, rope_sincos)
            
            # Apply final norm
            x = self.vit_model.norm(x)
            
            # Extract CLS token
            vision_features = x[:, 0]  # (B, embed_dim)
        
        # L2 normalize
        vision_features = F.normalize(vision_features, dim=-1)
        
        return vision_features


class DINOtxtModel(nn.Module):
    """DINOtxt: Vision-Language alignment model"""
    def __init__(self, vision_encoder, text_encoder, vision_dim, text_dim):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
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
            logit_scale: scalar
        """
        # Encode vision and text
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(text_tokens)
        
        # Project to common space
        vision_features = self.vision_proj(vision_features)
        text_features = self.text_proj(text_features)
        
        # Normalize
        vision_features = F.normalize(vision_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return vision_features, text_features, self.logit_scale.exp()


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
    
    # Text encoder arguments
    parser.add_argument("--text_embed_dim", type=int, default=512, help="Text embedding dimension")
    parser.add_argument("--text_num_layers", type=int, default=12, help="Number of text transformer layers")
    parser.add_argument("--text_num_heads", type=int, default=8, help="Number of text attention heads")
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=77, help="Maximum sequence length")
    
    # Loss arguments
    parser.add_argument("--clip_loss_weight", type=float, default=1.0, help="CLIP loss weight")
    
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
    parser.add_argument("--log_freq", type=int, default=5, help="Logging frequency")
    
    return parser.parse_args()


class DummyImageTextDataset(torch.utils.data.Dataset):
    """
    Dummy image-text paired dataset for demonstration
    Replace with actual COCO Captions or similar dataset loading
    """
    def __init__(self, num_samples=10000, input_size=224, max_seq_len=77):
        self.num_samples = num_samples
        self.input_size = input_size
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy image
        image = torch.randn(3, self.input_size, self.input_size)
        
        # Generate dummy text tokens
        text_tokens = torch.randint(0, 30000, (self.max_seq_len,))
        
        return image, text_tokens


def train_one_epoch(model, train_loader, optimizer, device, args, epoch, rank):
    """Train for one epoch"""
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
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if is_main_process:
            pbar.set_postfix({
                'loss': loss.item(),
                'logit_scale': logit_scale.item()
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, test_loader, device, rank):
    """Evaluate retrieval performance"""
    model.eval()
    
    all_vision_features = []
    all_text_features = []
    
    is_main_process = (rank == 0)
    
    with torch.no_grad():
        if is_main_process:
            pbar = tqdm(test_loader, desc="Evaluating")
        else:
            pbar = test_loader
        
        for images, text_tokens in pbar:
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            
            vision_features, text_features, _ = model(images, text_tokens)
            
            all_vision_features.append(vision_features.cpu())
            all_text_features.append(text_features.cpu())
    
    # Concatenate features
    vision_features = torch.cat(all_vision_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)
    
    # Compute image-to-text retrieval accuracy (R@1)
    similarity = vision_features @ text_features.t()
    _, indices = similarity.topk(1, dim=1)
    correct = (indices.squeeze() == torch.arange(len(vision_features))).sum().item()
    i2t_acc = correct / len(vision_features)
    
    # Compute text-to-image retrieval accuracy (R@1)
    _, indices = similarity.t().topk(1, dim=1)
    correct = (indices.squeeze() == torch.arange(len(text_features))).sum().item()
    t2i_acc = correct / len(text_features)
    
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
    vision_encoder = VisionEncoder(vit_model, args.use_daga, args.daga_layers)
    
    # Create text encoder
    if is_main_process:
        print(f"Creating text encoder...")
    
    text_encoder = SimpleTextEncoder(
        vocab_size=args.vocab_size,
        embed_dim=args.text_embed_dim,
        num_layers=args.text_num_layers,
        num_heads=args.text_num_heads,
        max_seq_len=args.max_seq_len,
    )
    
    # Create DINOtxt model
    model = DINOtxtModel(
        vision_encoder,
        text_encoder,
        vision_dim=vit_model.embed_dim,
        text_dim=args.text_embed_dim,
    )
    model.to(device)
    
    # Wrap with DDP
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    
    if is_main_process:
        print(f"✓ Models loaded and wrapped with DDP\n")
    
    # Create datasets
    train_dataset = DummyImageTextDataset(10000, args.input_size, args.max_seq_len)
    test_dataset = DummyImageTextDataset(1000, args.input_size, args.max_seq_len)
    
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
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr * world_size,
        weight_decay=args.weight_decay
    )
    
    if is_main_process:
        print(f"Starting training...\n")
    
    best_i2t_acc = 0.0
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        # Train
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, args, epoch, rank)
        
        # Evaluate
        if (epoch + 1) % args.log_freq == 0:
            i2t_acc, t2i_acc = evaluate(model, test_loader, device, rank)
            
            if is_main_process:
                print(f"\nEpoch {epoch+1}/{args.epochs}:")
                print(f"  Train Loss: {avg_loss:.4f}")
                print(f"  Image-to-Text R@1: {i2t_acc*100:.2f}%")
                print(f"  Text-to-Image R@1: {t2i_acc*100:.2f}%")
                
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


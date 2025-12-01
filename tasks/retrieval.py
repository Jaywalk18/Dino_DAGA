"""
Instance Retrieval Task with DINOv3 and DAGA
Evaluates on Revisited Oxford and Paris datasets
"""
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
import swanlab

from core.daga import DAGA
from core.backbones import get_attention_map, compute_daga_guidance_map, process_attention_weights
from core.utils import get_base_model


class RetrievalModel(nn.Module):
    """DINOv3 model for instance retrieval with optional DAGA"""
    
    def __init__(
        self,
        pretrained_vit,
        use_daga=False,
        daga_layers=[11],
        pooling='gem',
        gem_p=3.0,
    ):
        """
        Args:
            pretrained_vit: Pretrained DINOv3 backbone
            use_daga: Whether to use DAGA
            daga_layers: Layers to apply DAGA
            pooling: Pooling method ('cls', 'avg', 'gem')
            gem_p: GeM pooling power parameter
        """
        super().__init__()
        self.vit = pretrained_vit
        self.use_daga = use_daga
        self.daga_layers = daga_layers
        self.feature_dim = self.vit.embed_dim
        self.pooling = pooling
        self.gem_p = nn.Parameter(torch.ones(1) * gem_p)
        self.daga_guidance_layer_idx = len(self.vit.blocks) - 1
        
        self.num_storage_tokens = -1
        
        # Freeze backbone
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Initialize DAGA modules
        if self.use_daga:
            self.daga_modules = nn.ModuleDict(
                {str(i): DAGA(feature_dim=self.feature_dim) for i in daga_layers}
            )
            for param in self.daga_modules.parameters():
                param.requires_grad = True
        
        print(
            f"✓ RetrievalModel initialized:\n"
            f"  - Feature dim: {self.feature_dim}\n"
            f"  - Pooling: {pooling}\n"
            f"  - Use DAGA: {self.use_daga} (Layers: {self.daga_layers if self.use_daga else 'N/A'})"
        )
    
    def gem_pooling(self, x, eps=1e-6):
        """Generalized Mean Pooling"""
        return x.clamp(min=eps).pow(self.gem_p).mean(dim=1).pow(1.0 / self.gem_p)
    
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
        
        if self.use_daga:
            daga_guidance_map = compute_daga_guidance_map(
                self.vit, x_processed, H, W, self.daga_guidance_layer_idx
            )
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            if request_visualization_maps and idx == self.daga_guidance_layer_idx:
                with torch.no_grad():
                    adapted_attn_weights = get_attention_map(block, x_processed)
            
            x_processed = block(x_processed, rope_sincos)
            
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
        
        x_normalized = self.vit.norm(x_processed)
        
        # Extract features based on pooling method
        if self.pooling == 'cls':
            features = x_normalized[:, 0]  # CLS token
        elif self.pooling == 'avg':
            patch_tokens = x_normalized[:, 1 + num_registers:]
            features = patch_tokens.mean(dim=1)
        elif self.pooling == 'gem':
            patch_tokens = x_normalized[:, 1 + num_registers:]
            features = self.gem_pooling(patch_tokens)
        else:
            features = x_normalized[:, 0]
        
        # L2 normalize
        features = F.normalize(features, p=2, dim=1)
        
        return features, adapted_attn_weights, daga_guidance_map


def extract_features(model, dataloader, device):
    """Extract features from all images in dataloader"""
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features, _, _ = model(images, False)
            features_list.append(features.cpu())
    
    return torch.cat(features_list, dim=0)


def extract_query_features(model, query_dataset, device, batch_size=32):
    """Extract features from query images"""
    model.eval()
    features_list = []
    
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for images, _ in tqdm(query_loader, desc="Extracting query features"):
            images = images.to(device)
            features, _, _ = model(images, False)
            features_list.append(features.cpu())
    
    return torch.cat(features_list, dim=0)


def compute_ap(ranks, num_positives):
    """Compute Average Precision"""
    if num_positives == 0:
        return 0.0
    
    ap = 0.0
    for i, rank in enumerate(ranks):
        precision_at_i = (i + 1) / (rank + 1)
        ap += precision_at_i
    
    ap /= num_positives
    return ap


def evaluate_retrieval(query_features, db_features, dataset, protocol='medium'):
    """
    Evaluate retrieval performance
    
    Args:
        query_features: (num_queries, dim) tensor
        db_features: (num_db, dim) tensor
        dataset: RevisitedOxfordParisDataset instance
        protocol: 'easy', 'medium', or 'hard'
    
    Returns:
        mAP, recalls dict
    """
    num_queries = query_features.shape[0]
    
    # Compute similarity
    similarity = torch.mm(query_features, db_features.t())
    
    # Get rankings
    _, rankings = torch.sort(similarity, dim=1, descending=True)
    rankings = rankings.cpu().numpy()
    
    aps = []
    recalls_at_k = {1: [], 5: [], 10: [], 100: []}
    
    for q_idx in range(num_queries):
        easy, hard, junk = dataset.get_query_ground_truth(q_idx)
        
        # Define positives based on protocol
        if protocol == 'easy':
            positives = set(easy)
        elif protocol == 'medium':
            positives = set(easy + hard)
        else:  # hard
            positives = set(hard)
        
        junk_set = set(junk)
        
        if len(positives) == 0:
            continue
        
        # Filter junk from rankings
        filtered_ranks = []
        pos_ranks = []
        
        for rank, db_idx in enumerate(rankings[q_idx]):
            if db_idx in junk_set:
                continue
            
            if db_idx in positives:
                pos_ranks.append(len(filtered_ranks))
            
            filtered_ranks.append(db_idx)
        
        # Compute AP
        ap = compute_ap(pos_ranks, len(positives))
        aps.append(ap)
        
        # Compute recall@k
        for k in recalls_at_k.keys():
            top_k = set(filtered_ranks[:k])
            recall = len(top_k & positives) / len(positives)
            recalls_at_k[k].append(recall)
    
    mAP = np.mean(aps) * 100 if aps else 0.0
    mean_recalls = {k: np.mean(v) * 100 for k, v in recalls_at_k.items()}
    
    return mAP, mean_recalls


def run_retrieval_evaluation(
    model,
    db_dataset,
    query_dataset,
    device,
    args,
    output_dir,
    rank=0,
):
    """Run complete retrieval evaluation"""
    is_main_process = (rank == 0)
    
    if is_main_process:
        print(f"\n{'='*70}")
        print("Extracting database features...")
    
    db_loader = DataLoader(
        db_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    db_features = extract_features(model, db_loader, device)
    
    if is_main_process:
        print(f"✓ Database features: {db_features.shape}")
        print("\nExtracting query features...")
    
    query_features = extract_query_features(model, query_dataset, device, args.batch_size)
    
    if is_main_process:
        print(f"✓ Query features: {query_features.shape}")
    
    # Evaluate on all protocols
    results = {}
    for protocol in ['easy', 'medium', 'hard']:
        mAP, recalls = evaluate_retrieval(query_features, db_features, db_dataset, protocol)
        results[protocol] = {'mAP': mAP, 'recalls': recalls}
        
        if is_main_process:
            print(f"\n📊 {protocol.upper()} Protocol:")
            print(f"   mAP: {mAP:.2f}%")
            print(f"   R@1: {recalls[1]:.2f}% | R@5: {recalls[5]:.2f}% | R@10: {recalls[10]:.2f}%")
    
    # Log to SwanLab
    if is_main_process and getattr(args, 'enable_swanlab', True):
        log_dict = {
            'mAP_easy': results['easy']['mAP'],
            'mAP_medium': results['medium']['mAP'],
            'mAP_hard': results['hard']['mAP'],
            'R@1_medium': results['medium']['recalls'][1],
            'R@5_medium': results['medium']['recalls'][5],
            'R@10_medium': results['medium']['recalls'][10],
        }
        swanlab.log(log_dict)
    
    return results


def visualize_retrieval_results(
    model, db_dataset, query_dataset, args, output_dir, num_queries=5, top_k=5
):
    """Visualize retrieval results"""
    device = next(model.parameters()).device
    base_model = get_base_model(model)
    base_model.eval()
    
    # Extract features
    db_loader = DataLoader(db_dataset, batch_size=32, shuffle=False)
    db_features = extract_features(model, db_loader, device)
    query_features = extract_query_features(model, query_dataset, device, 32)
    
    # Compute similarity
    similarity = torch.mm(query_features, db_features.t())
    _, rankings = torch.sort(similarity, dim=1, descending=True)
    
    vis_save_path = Path(output_dir) / "visualizations"
    vis_save_path.mkdir(parents=True, exist_ok=True)
    
    vis_figs = []
    
    for q_idx in range(min(num_queries, len(query_dataset))):
        fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))
        fig.suptitle(f"Query {q_idx} Retrieval Results", fontsize=14, fontweight='bold')
        
        # Query image
        q_img, _ = query_dataset[q_idx]
        q_img_np = q_img.cpu().numpy().transpose(1, 2, 0)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        q_img_np = np.clip(std * q_img_np + mean, 0, 1)
        
        axes[0].imshow(q_img_np)
        axes[0].set_title("Query", fontsize=10)
        axes[0].axis('off')
        
        # Top-k results
        easy, hard, junk = db_dataset.get_query_ground_truth(q_idx)
        positives = set(easy + hard)
        
        for k in range(top_k):
            db_idx = rankings[q_idx, k].item()
            db_img, _ = db_dataset[db_idx]
            db_img_np = db_img.cpu().numpy().transpose(1, 2, 0)
            db_img_np = np.clip(std * db_img_np + mean, 0, 1)
            
            axes[k + 1].imshow(db_img_np)
            
            # Mark positive/negative
            is_positive = db_idx in positives
            color = 'green' if is_positive else 'red'
            title = f"#{k+1}" + (" ✓" if is_positive else " ✗")
            axes[k + 1].set_title(title, fontsize=10, color=color)
            axes[k + 1].axis('off')
        
        plt.tight_layout()
        vis_figs.append(fig)
        
        fig.savefig(vis_save_path / f"query_{q_idx}_results.png", dpi=100)
        plt.close(fig)
    
    return vis_figs


def setup_training_components(model, args):
    """Setup optimizer for DAGA training (if applicable)"""
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    base_model = model.module if isinstance(model, (DataParallel, DDP)) else model
    
    if not base_model.use_daga:
        # No training needed for baseline
        return None, None
    
    daga_params = []
    for name, param in base_model.named_parameters():
        if param.requires_grad and "daga" in name:
            daga_params.append(param)
    
    if not daga_params:
        return None, None
    
    optimizer = torch.optim.AdamW(
        daga_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    return optimizer, scheduler


def train_with_contrastive_loss(
    model, db_dataset, query_dataset, optimizer, scheduler, device, args, output_dir, rank=0
):
    """
    Train DAGA with contrastive loss for retrieval.
    Uses query-positive pairs from ground truth.
    """
    is_main_process = (rank == 0)
    
    if optimizer is None:
        if is_main_process:
            print("No DAGA modules to train. Running evaluation only.")
        return
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Sample query-positive pairs
        num_queries = query_dataset.retrieval_dataset.get_num_queries()
        
        for q_idx in range(num_queries):
            easy, hard, _ = query_dataset.retrieval_dataset.get_query_ground_truth(q_idx)
            positives = easy + hard
            
            if len(positives) == 0:
                continue
            
            # Get query
            q_img, _ = query_dataset[q_idx]
            q_img = q_img.unsqueeze(0).to(device)
            
            # Sample a positive
            pos_idx = np.random.choice(positives)
            pos_img, _ = db_dataset[pos_idx]
            pos_img = pos_img.unsqueeze(0).to(device)
            
            # Sample a negative
            all_indices = set(range(len(db_dataset)))
            negatives = list(all_indices - set(positives))
            neg_idx = np.random.choice(negatives)
            neg_img, _ = db_dataset[neg_idx]
            neg_img = neg_img.unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            
            # Extract features
            q_feat, _, _ = model(q_img, False)
            pos_feat, _, _ = model(pos_img, False)
            neg_feat, _, _ = model(neg_img, False)
            
            # Triplet loss
            pos_dist = (q_feat - pos_feat).pow(2).sum()
            neg_dist = (q_feat - neg_feat).pow(2).sum()
            margin = 0.5
            loss = F.relu(pos_dist - neg_dist + margin)
            
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if is_main_process:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
            
            if getattr(args, 'enable_swanlab', True):
                swanlab.log({'train_loss': avg_loss, 'epoch': epoch + 1})


import torch
import torch.nn as nn
import os


def load_dinov3_backbone(model_name, pretrained_path, dinov3_repo_path=None):
    """
    Load DINOv3 backbone model
    """
    if dinov3_repo_path is None:
        dinov3_repo_path = '/home/user/zhoutianjian/dinov3'
    
    # Handle absolute vs relative paths
    if os.path.isabs(pretrained_path):
        weights_path = pretrained_path
    else:
        weights_path = os.path.join('/home/user/zhoutianjian/DAGA/checkpoints', pretrained_path)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    
    import torch.distributed as dist
    
    # Check if this is a training checkpoint (contains model_state_dict) or a plain state_dict
    # Only rank 0 extracts to avoid file conflicts
    actual_weights_path = weights_path + '.extracted'
    if not os.path.exists(actual_weights_path):
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # This is a training checkpoint, extract and clean the model weights
                state_dict = checkpoint['model_state_dict']
                # Remove 'module.' and 'vit.' prefixes (from DDP and classification wrapper)
                cleaned_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key
                    # Remove 'module.' prefix (from DDP)
                    if new_key.startswith('module.'):
                        new_key = new_key[7:]
                    # Remove 'vit.' prefix (from classification wrapper)
                    if new_key.startswith('vit.'):
                        new_key = new_key[4:]
                    
                    # Skip classifier and daga_modules layers (only keep backbone)
                    if new_key.startswith('classifier.') or new_key.startswith('daga_modules.'):
                        continue
                    
                    cleaned_state_dict[new_key] = value
                torch.save(cleaned_state_dict, actual_weights_path, _use_new_zipfile_serialization=True)
                del checkpoint, state_dict, cleaned_state_dict  # Free memory
            else:
                # Plain state_dict, use original path
                actual_weights_path = weights_path
        # Wait for rank 0 to finish extraction
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    
    # Use extracted weights if they exist, otherwise use original
    if os.path.exists(actual_weights_path) and actual_weights_path != weights_path:
        weights_path = actual_weights_path
    
    # Only print from rank 0 or if not in distributed mode
    should_print = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    
    if should_print:
        print(f"✓ Loading checkpoint from: {weights_path}")
    
    # In DDP environment, make model loading sequential to avoid race conditions
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Load models sequentially by rank to avoid file system contention
        for r in range(world_size):
            if rank == r:
                vit_model = torch.hub.load(
                    dinov3_repo_path,
                    model_name,
                    source='local',
                    weights=weights_path
                )
            # Wait for current rank to finish before next rank starts
            dist.barrier()
    else:
        vit_model = torch.hub.load(
            dinov3_repo_path,
            model_name,
            source='local',
            weights=weights_path
        )
    
    if should_print:
        print(f"\n[DEBUG] Model loaded:")
        print(f"  embed_dim: {vit_model.embed_dim}")
        print(f"  n_storage_tokens: {getattr(vit_model, 'n_storage_tokens', 0)}")
        if hasattr(vit_model, "storage_tokens"):
            print(f"  storage_tokens shape: {vit_model.storage_tokens.shape}")
        print(f"  cls_token shape: {vit_model.cls_token.shape}")
    
    return vit_model


def get_attention_map(block, x):
    """
    Helper function to calculate attention maps from a transformer block.
    Applies LayerNorm before calculating QKV.
    """
    attn_module = block.attn
    normed_x = block.norm1(x)
    
    B, N, C = normed_x.shape
    num_heads = attn_module.num_heads
    head_dim = C // num_heads
    
    qkv = attn_module.qkv(normed_x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    
    q = q * attn_module.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    
    return attn


def process_attention_weights(raw_weights, num_patches_expected, H, W):
    """
    Process raw attention weights to extract patch attention.
    Assumes token order: [CLS], [REG], [PATCH]
    """
    B, _, seq_len, _ = raw_weights.shape
    
    num_registers = seq_len - num_patches_expected - 1
    if num_registers < 0:
        print(f"⚠ Warning: Negative registers ({num_registers}). Assuming 0.")
        num_registers = 0
    
    cls_attn_all_tokens = raw_weights[:, :, 0, 1:]
    patch_start_index = num_registers
    cls_attn_patch_tokens_headed = cls_attn_all_tokens[:, :, patch_start_index:]
    cls_attn_patches = cls_attn_patch_tokens_headed.mean(dim=1)
    
    min_val = cls_attn_patches.amin(dim=1, keepdim=True)
    max_val = cls_attn_patches.amax(dim=1, keepdim=True)
    cls_attn_normalized = (cls_attn_patches - min_val) / (max_val - min_val + 1e-8)
    
    B, N = cls_attn_normalized.shape
    
    if N != num_patches_expected:
        print(f"⚠ Warning: Attention patch count mismatch! Expected {num_patches_expected}, got {N}.")
        if N > 0 and int(N**0.5) * int(N**0.5) == N:
            H = W = int(N**0.5)
        else:
            return None
    
    if H > 0 and W > 0:
        return cls_attn_normalized.reshape(B, H, W).cpu().numpy()
    else:
        return None


def compute_daga_guidance_map(vit_model, x_processed, H, W, guidance_layer_idx):
    """
    Compute DAGA guidance map from frozen backbone attention.
    """
    B, seq_len, C = x_processed.shape
    num_patches = H * W
    num_registers = seq_len - num_patches - 1
    
    captured_guidance_attn = None
    
    with torch.no_grad():
        guidance_features = x_processed.clone()
        
        for i in range(len(vit_model.blocks)):
            rope_sincos = vit_model.rope_embed(H=H, W=W) if vit_model.rope_embed else None
            
            if i == guidance_layer_idx:
                attn_weights = get_attention_map(
                    vit_model.blocks[i], 
                    guidance_features
                )
                captured_guidance_attn = attn_weights
            
            guidance_features = vit_model.blocks[i](guidance_features, rope_sincos)
    
    if captured_guidance_attn is not None:
        # Get attention from CLS (idx 0) to all other tokens (idx 1:)
        cls_attn_all_tokens = captured_guidance_attn[:, :, 0, 1:]
        
        # Skip register tokens to get only patch attention
        patch_start_index = num_registers
        cls_attn_patch_tokens_headed = cls_attn_all_tokens[:, :, patch_start_index:]
        
        # Average over heads
        cls_attn_patches = cls_attn_patch_tokens_headed.mean(dim=1)
        
        # Normalize for guidance map
        min_val = cls_attn_patches.amin(dim=1, keepdim=True)
        max_val = cls_attn_patches.amax(dim=1, keepdim=True)
        cls_attn = (cls_attn_patches - min_val) / (max_val - min_val + 1e-8)
        
        if cls_attn.shape[1] == num_patches:
            return cls_attn.reshape(B, H, W)
    
    return None

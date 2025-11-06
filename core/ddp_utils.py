"""
DDP utilities for distributed training
"""
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_ddp():
    """Initialize DDP environment
    
    Using gloo backend instead of nccl to avoid SIGSEGV issues.
    Models and data still run on GPU, only communication uses gloo.
    """
    # Initialize process group with gloo backend
    import datetime
    
    backend = 'gloo'  # Use gloo to avoid NCCL SIGSEGV issues
    
    dist.init_process_group(
        backend=backend, 
        init_method='env://',
        timeout=datetime.timedelta(seconds=300)
    )
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set CUDA device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
    
    # Synchronize all processes
    if dist.is_initialized():
        dist.barrier()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"✓ DDP initialized with backend: {backend}")
        print(f"  Using CUDA device: {local_rank}" if torch.cuda.is_available() else "  Using CPU")
    
    return local_rank, rank, world_size


def cleanup_ddp():
    """Cleanup DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_ddp_dataloaders(train_dataset, val_dataset, batch_size, world_size, rank, num_workers=4, collate_fn=None):
    """Create dataloaders with DistributedSampler for DDP
    
    Note: num_workers=0 for DDP to avoid SIGSEGV errors with multiprocessing.
    DDP already provides parallelism across GPUs, so single-process loading per GPU is sufficient.
    """
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, val_loader


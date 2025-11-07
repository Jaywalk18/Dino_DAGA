# compare_models.py
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import torchvision.datasets as datasets

# ✨ MODIFICATION: Import distributed modules
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# ✨ FIX: Import tqdm for the progress bar
from tqdm import tqdm

# Assuming other imports like ModifiedViT and vits are correct
try:
    from dino_finetune_experiment import ModifiedViT, ImageNetValDataset
except ImportError:
    print("❌ Error: Could not import 'ModifiedViT' or 'ImageNetValDataset'.")
    print("   Please ensure 'dino_finetune_experiment.py' is in the same directory or accessible.")
    exit(1)
import vision_transformer as vits

warnings.filterwarnings("ignore")


def setup_for_distributed(is_master):
    """Disable printing when not in master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode():
    """Initialize the distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(gpu)
        setup_for_distributed(rank == 0)
        print(f"🌍 Distributed mode initialized. Rank {rank}/{world_size} on GPU {gpu}")
    else:
        print("⚪ Not in distributed mode.")
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'


def is_dist_avail_and_initialized():
    """Check if distributed mode is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get the rank of the current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare Predictions and Attention of Two ViT Models")
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        required=True,
        help="Path to the baseline model's .pth checkpoint.",
    )
    parser.add_argument(
        "--plugin_model_path",
        type=str,
        required=True,
        help="Path to the plugin (e.g., DAGA) model's .pth checkpoint.",
    )
    # MODIFIED: Added 'imagenet' to choices
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "imagenet100", "imagenet"],
        default="cifar100",
        help="Dataset to use for comparison.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the root directory for the dataset.",
    )
    parser.add_argument("--input_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference PER GPU.")
    parser.add_argument(
        "--num_top_diff",
        type=int,
        default=10,
        help="Number of images with the largest prediction difference to visualize.",
    )
    parser.add_argument(
        "--output_dir",
        default="./comparison_outputs",
        help="Directory to save comparison visualizations.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=8,
        help="Patch size of ViT (default: 8 for ViT-S).",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=384,
        help="Feature dimension of ViT (default: 384 for ViT-S).",
    )
    return parser.parse_args()


def load_model_from_checkpoint(cli_args, checkpoint_path, device):
    """Loads a model from a checkpoint file, handling different saved configurations."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Use args from checkpoint if available, otherwise fallback to CLI args and heuristics
    if "args" in checkpoint and checkpoint["args"]:
        args_from_checkpoint = argparse.Namespace(**checkpoint["args"])
        print(f"✓ Found 'args' in checkpoint '{Path(checkpoint_path).name}'.")
    else:
        print(f"⚠ Warning: 'args' not found in checkpoint '{Path(checkpoint_path).name}'.")
        print("   Will rely on command-line arguments and heuristics for model construction.")
        args_from_checkpoint = argparse.Namespace(
            use_daga=False,
            daga_layers=[],
            dataset=cli_args.dataset,
            patch_size=cli_args.patch_size,
            feature_dim=cli_args.feature_dim,
            vis_attn_layer=11,
            enable_visualization=True,
        )
        # Heuristic to detect DAGA if not specified in args
        if any("daga_module" in k for k in checkpoint["model_state_dict"]):
            args_from_checkpoint.use_daga = True
            print("   -> Heuristic: Detected 'daga_module' keys. Setting use_daga=True.")
    
    # Determine number of classes
    if args_from_checkpoint.dataset == "cifar10":
        num_classes = 10
    elif args_from_checkpoint.dataset == "cifar100":
        num_classes = 100
    elif args_from_checkpoint.dataset == "imagenet100":
        num_classes = 100
    # MODIFIED: Added 'imagenet' case
    elif args_from_checkpoint.dataset == "imagenet":
        # Full ImageNet has 1000 classes
        num_classes = 1000
    else:
        raise ValueError(f"Unsupported dataset in checkpoint args: {args_from_checkpoint.dataset}")

    print(f"Loading model from: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"  - Original Epoch: {checkpoint.get('epoch', 'N/A')}, Best Acc: {checkpoint.get('best_acc', 'N/A'):.2f}%")

    vit_model = vits.__dict__["vit_small"](patch_size=args_from_checkpoint.patch_size, num_classes=0)
    
    model = ModifiedViT(
        pretrained_vit=vit_model,
        num_classes=num_classes,
        use_daga=getattr(args_from_checkpoint, "use_daga", False),
        daga_layers=getattr(args_from_checkpoint, "daga_layers", []),
        feature_dim=getattr(args_from_checkpoint, "feature_dim", cli_args.feature_dim),
        enable_visualization=True, # Always enable for comparison
        vis_attn_layer=getattr(args_from_checkpoint, "vis_attn_layer", 11),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("✓ Model loaded and configured successfully.")
    return model


def process_raw_weights(raw_weights):
    """Processes raw attention weights into a 2D heatmap."""
    if raw_weights is None:
        return None
    # Average attention weights across heads for the [CLS] token
    cls_attn = raw_weights[:, :, 0, 1:].mean(dim=1)
    B, N = cls_attn.shape
    h = w = int(N**0.5)
    return cls_attn.reshape(B, h, w).cpu().numpy()


def find_differing_predictions_and_heatmaps(model1, model2, dataloader, device):
    """
    Finds differing predictions on the local GPU's subset of data.
    Returns a dictionary of results for aggregation.
    """
    all_diffs, all_images, all_labels = [], [], []
    all_heatmaps1, all_heatmaps2 = [], []
    all_preds1, all_preds2 = [], []

    desc = f"Comparing Models (GPU {os.environ['LOCAL_RANK']})"
    
    with torch.no_grad():
        for images, labels_from_loader in tqdm(dataloader, desc=desc, disable=(get_rank() != 0)):
            images = images.to(device)
            labels = labels_from_loader.to(device)

            # Get predictions and attention from the first model
            logits1, attn_weights1, _ = model1(images, request_visualization_maps=True)
            probs1 = F.softmax(logits1, dim=1)
            preds1 = logits1.argmax(dim=1)
            heatmap1 = process_raw_weights(attn_weights1)

            # Get predictions and attention from the second model
            logits2, attn_weights2, _ = model2(images, request_visualization_maps=True)
            probs2 = F.softmax(logits2, dim=1)
            preds2 = logits2.argmax(dim=1)
            heatmap2 = process_raw_weights(attn_weights2)
            
            # Find cases where model1 is wrong and model2 is correct
            if heatmap1 is not None and heatmap2 is not None:
                mask = (preds1 != labels) & (preds2 == labels)
                
                if mask.sum() > 0:
                    gt_labels_masked = labels[mask].unsqueeze(1)
                    # Probability difference on the ground truth class
                    gt_probs1 = probs1[mask].gather(1, gt_labels_masked)
                    gt_probs2 = probs2[mask].gather(1, gt_labels_masked)
                    diff = (gt_probs2 - gt_probs1).view(-1)

                    # Store all relevant data for these differing samples
                    all_diffs.append(diff.cpu())
                    all_images.append(images[mask].cpu())
                    all_labels.append(labels[mask].cpu())
                    all_preds1.append(preds1[mask].cpu())
                    all_preds2.append(preds2[mask].cpu())
                    
                    mask_cpu_np = mask.cpu().numpy()
                    all_heatmaps1.append(heatmap1[mask_cpu_np])
                    all_heatmaps2.append(heatmap2[mask_cpu_np])

    if not all_diffs:
        return None

    # Consolidate results from all batches on this GPU
    local_results = {
        'images': torch.cat(all_images), 'labels': torch.cat(all_labels),
        'preds1': torch.cat(all_preds1), 'preds2': torch.cat(all_preds2),
        'heatmaps1': np.concatenate(all_heatmaps1, axis=0),
        'heatmaps2': np.concatenate(all_heatmaps2, axis=0),
        'diffs': torch.cat(all_diffs)
    }
    return local_results


def visualize_top_differences(top_results, output_dir, class_names):
    """Generates and saves visualizations for the top differing images."""
    print(f"🖼️  Generating visualizations for top {len(top_results['images'])} differing images...")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_np = top_results["images"].numpy()
    
    for i in range(len(images_np)):
        true_label = class_names[top_results["labels"][i]]
        pred1_label = class_names[top_results["preds1"][i]]
        pred2_label = class_names[top_results["preds2"][i]]
        diff_score = top_results["diffs"][i].item()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        is_correct1 = "✓" if true_label == pred1_label else "✗"
        is_correct2 = "✓" if true_label == pred2_label else "✗"
        fig.suptitle(
            f"Image #{i+1} | Prediction Disagreement (Score: {diff_score:.3f})\nTrue Label: {true_label}",
            fontsize=16,
            y=1.02,
        )
        
        # Denormalize and display the original image
        img = images_np[i].transpose(1, 2, 0)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img = np.clip(std * img + mean, 0, 1)
        axes[0].imshow(img)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")
        
        # Display baseline model's attention
        axes[1].imshow(img)
        axes[1].imshow(top_results["heatmaps1"][i], cmap="viridis", alpha=0.5)
        axes[1].set_title(
            f"Baseline Pred: {pred1_label} {is_correct1}\n(Attention Heatmap)",
            fontsize=12,
        )
        axes[1].axis("off")
        
        # Display plugin model's attention
        axes[2].imshow(img)
        axes[2].imshow(top_results["heatmaps2"][i], cmap="viridis", alpha=0.5)
        axes[2].set_title(
            f"DAGA-Plugin Pred: {pred2_label} {is_correct2}\n(Attention Heatmap)",
            fontsize=12,
        )
        axes[2].axis("off")
        
        plt.tight_layout()
        safe_true_label = true_label.replace("/", "_")
        save_path = output_dir / f"comparison_{i+1}_label_{safe_true_label}_diff_{diff_score:.3f}.png"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        
    print(f"✓ Visualizations saved to: {output_dir}")


def main():
    init_distributed_mode()
    
    args = parse_arguments()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # MODIFIED: Refactored dataset loading to include 'imagenet'
    if args.dataset == "cifar10":
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
    elif args.dataset == "cifar100":
        test_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=test_transform)
    elif args.dataset == "imagenet100":
        val_path = os.path.join(args.data_path, "val")
        test_dataset = datasets.ImageFolder(val_path, transform=test_transform)
    elif args.dataset == "imagenet":
        val_path = os.path.join(args.data_path, "val")
        val_annot_path = os.path.join(args.data_path, "val_annotations.txt")
        if not os.path.exists(val_annot_path):
            raise FileNotFoundError(f"ImageNet validation annotation file not found: {val_annot_path}")
        test_dataset = ImageNetValDataset(val_path, val_annot_path, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    model1_baseline = load_model_from_checkpoint(args, args.baseline_model_path, device)
    model2_plugin = load_model_from_checkpoint(args, args.plugin_model_path, device)

    print("\n🔍 Starting inference and comparison...")
    local_results = find_differing_predictions_and_heatmaps(
        model1_baseline, model2_plugin, test_loader, device
    )

    # Gather results from all GPUs to the main process
    if is_dist_avail_and_initialized():
        all_gathered_results = [None] * dist.get_world_size()
        dist.gather_object(
            local_results,
            all_gathered_results if get_rank() == 0 else None,
            dst=0
        )
    else:
        all_gathered_results = [local_results]

    # Post-processing on the main process
    if get_rank() == 0:
        print("\n✓ Inference complete on all GPUs. Analyzing results on main process...")
        
        valid_results = [res for res in all_gathered_results if res is not None]

        if not valid_results:
            print("⚠ No images found matching criteria across all GPUs. Exiting.")
            if is_dist_avail_and_initialized():
                dist.destroy_process_group()
            return

        # Combine results from all GPUs
        combined_results = {
            'images': torch.cat([res['images'] for res in valid_results]),
            'labels': torch.cat([res['labels'] for res in valid_results]),
            'preds1': torch.cat([res['preds1'] for res in valid_results]),
            'preds2': torch.cat([res['preds2'] for res in valid_results]),
            'heatmaps1': np.concatenate([res['heatmaps1'] for res in valid_results], axis=0),
            'heatmaps2': np.concatenate([res['heatmaps2'] for res in valid_results], axis=0),
            'diffs': torch.cat([res['diffs'] for res in valid_results])
        }
        
        print(f"Total of {len(combined_results['diffs'])} images where baseline failed and plugin succeeded were found.")

        # Sort by the difference score and select the top N
        sorted_indices = torch.argsort(combined_results['diffs'], descending=True)
        top_n = min(args.num_top_diff, len(sorted_indices))
        top_indices = sorted_indices[:top_n]

        top_results = {key: val[top_indices] for key, val in combined_results.items()}

        visualize_top_differences(top_results, Path(args.output_dir), test_dataset.classes)

    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


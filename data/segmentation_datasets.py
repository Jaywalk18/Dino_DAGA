import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image


def get_segmentation_dataset(args):
    """
    Load and prepare segmentation dataset (ADE20K, COCO)
    """
    if args.dataset == "ade20k":
        return get_ade20k_dataset(args)
    elif args.dataset == "coco":
        return get_coco_seg_dataset(args)
    else:
        raise ValueError(f"Unknown segmentation dataset: {args.dataset}")


def get_ade20k_dataset(args):
    """Load ADE20K dataset for semantic segmentation"""
    
    train_img_dir = os.path.join(args.data_path, "images/ADE/training")
    val_img_dir = os.path.join(args.data_path, "images/ADE/validation")
    train_ann_dir = os.path.join(args.data_path, "images/ADE/training")  # Annotations are in same dir
    val_ann_dir = os.path.join(args.data_path, "images/ADE/validation")
    
    # Verify paths exist
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"ADE20K train images not found: {train_img_dir}")
    if not os.path.exists(val_img_dir):
        raise FileNotFoundError(f"ADE20K val images not found: {val_img_dir}")
    if not os.path.exists(train_ann_dir):
        raise FileNotFoundError(f"ADE20K train annotations not found: {train_ann_dir}")
    if not os.path.exists(val_ann_dir):
        raise FileNotFoundError(f"ADE20K val annotations not found: {val_ann_dir}")
    
    # Define transforms
    train_transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    mask_transform = T.Compose([
        T.Resize((args.input_size, args.input_size), interpolation=T.InterpolationMode.NEAREST),
    ])
    
    # Load datasets
    train_dataset = ADE20KDataset(
        img_dir=train_img_dir,
        ann_dir=train_ann_dir,
        transform=train_transform,
        mask_transform=mask_transform
    )
    
    val_dataset = ADE20KDataset(
        img_dir=val_img_dir,
        ann_dir=val_ann_dir,
        transform=val_transform,
        mask_transform=mask_transform
    )
    
    # ADE20K has 150 classes + 1 background
    num_classes = 150
    
    return train_dataset, val_dataset, num_classes


def get_coco_seg_dataset(args):
    """Load COCO dataset for semantic segmentation"""
    raise NotImplementedError("COCO segmentation dataset not implemented yet")


class ADE20KDataset(Dataset):
    """ADE20K Semantic Segmentation Dataset"""
    def __init__(self, img_dir, ann_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Recursively get all image files from subdirectories
        self.img_files = []
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                if f.endswith('.jpg'):
                    self.img_files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.img_files)
    
    def _validate_samples(self, num_samples=3):
        """Validate a few samples to check mask value ranges"""
        print(f"Validating {num_samples} samples...")
        for i in range(num_samples):
            img_path = self.img_files[i]
            mask_base = img_path.replace('.jpg', '')
            mask_path = os.path.join(mask_base, f"{os.path.basename(mask_base)}_seg.png")
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path), dtype=np.int64)
                unique_vals = np.unique(mask)
                print(f"  Sample {i}: mask values in range [{unique_vals.min()}, {unique_vals.max()}]")
                if unique_vals.max() > 150:
                    print(f"    Warning: Found value {unique_vals.max()} > 150")
            else:
                print(f"  Sample {i}: mask not found at {mask_path}")

    def __getitem__(self, idx):
        # Image path is already absolute from os.walk
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask (annotation) - _seg.png is in the same directory as .jpg, not in subfolder
        # Example: /path/to/ADE_val_00000749.jpg -> /path/to/ADE_val_00000749_seg.png
        mask_path = img_path.replace('.jpg', '_seg.png')
        
        if os.path.exists(mask_path):
            mask_img = Image.open(mask_path)
            # ADE20K masks are stored as RGB images, class index is in R channel
            mask_arr = np.array(mask_img, dtype=np.int64)
            if len(mask_arr.shape) == 3:  # RGB format
                mask = mask_arr[:, :, 0]  # Use R channel
            else:
                mask = mask_arr
            
            # ADE20K: 0 is background/unlabeled, 1-150 are object classes
            # Convert to: 255 is ignore (background), 0-149 are classes
            mask = mask.copy()
            mask[mask == 0] = 255  # Background becomes ignore index
            mask[mask != 255] -= 1  # Shift 1-150 to 0-149
        else:
            # If no mask found, create empty mask with ignore index
            print(f"Warning: Mask not found at {mask_path}")
            mask = np.full((image.size[1], image.size[0]), 255, dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            # Convert mask to PIL for transform, then back to tensor
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil = self.mask_transform(mask_pil)
            mask = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
        else:
            mask = torch.from_numpy(mask)
        
        return image, mask

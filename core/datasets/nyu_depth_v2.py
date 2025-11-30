"""
NYU Depth V2 Dataset Loader
Supports both:
1. BTS format (24K images) - recommended for training
2. Labeled mat format (1449 images) - fallback option
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import h5py
from PIL import Image
import torchvision.transforms as transforms


class NYUDepthV2Dataset(Dataset):
    """
    NYU Depth V2 Dataset - supports BTS format (24K images)
    
    BTS format structure:
    - nyu_train.txt / nyu_test.txt: index files
    - scene_name/rgb_xxxxx.jpg: RGB images
    - scene_name/sync_depth_xxxxx.png: depth maps (16-bit PNG, in mm)
    """
    
    def __init__(self, 
                 data_path,
                 split='train',
                 input_size=518,
                 min_depth=0.001,
                 max_depth=10.0,
                 augmentation=False,
                 sample_ratio=None):
        """
        Args:
            data_path: Path to NYU dataset directory (BTS format or labeled.mat format)
            split: 'train' or 'test'
            input_size: Target image size (default: 518 for DINOv3 depth)
            min_depth: Minimum depth value (meters)
            max_depth: Maximum depth value (meters)
            augmentation: Whether to apply data augmentation (train only)
            sample_ratio: Use only a fraction of the dataset (0.0 to 1.0), None = use all
        """
        super().__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.input_size = input_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.augmentation = augmentation and (split == 'train')
        self.sample_ratio = sample_ratio
        
        # Try BTS format first (recommended)
        self.use_bts_format = self._try_bts_format()
        
        if not self.use_bts_format:
            # Fallback to labeled.mat format
            self._init_mat_format()
        
        # Define transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Image transforms
        if self.augmentation:
            self.img_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
            self.depth_transform = transforms.Resize(
                (input_size, input_size), 
                interpolation=transforms.InterpolationMode.NEAREST
            )
        else:
            self.img_transform = transforms.Resize((input_size, input_size))
            self.depth_transform = transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.NEAREST
            )
    
    def _try_bts_format(self):
        """Try to initialize BTS format dataset"""
        # Check for BTS format in various possible locations
        possible_roots = [
            self.data_path / 'nyu',           # NYU_BTS/nyu/
            self.data_path,                    # Direct path
            self.data_path.parent / 'NYU_BTS' / 'nyu',  # Sibling directory
        ]
        
        for root in possible_roots:
            train_file = root / 'nyu_train.txt'
            test_file = root / 'nyu_test.txt'
            
            if train_file.exists() and test_file.exists():
                self.bts_root = root
                self._init_bts_format()
                return True
        
        return False
    
    def _init_bts_format(self):
        """Initialize BTS format dataset"""
        # Load index file
        if self.split == 'train':
            index_file = self.bts_root / 'nyu_train.txt'
        else:
            index_file = self.bts_root / 'nyu_test.txt'
        
        print(f"Loading NYU Depth V2 (BTS format) from {self.bts_root}")
        print(f"Index file: {index_file}")
        
        self.samples = []
        with open(index_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rgb_path = parts[0].lstrip('/')
                    depth_path = parts[1].lstrip('/')
                    # focal_length = float(parts[2]) if len(parts) > 2 else 518.8579
                    self.samples.append((rgb_path, depth_path))
        
        # Apply sample ratio
        if self.sample_ratio is not None and 0.0 < self.sample_ratio < 1.0:
            num_samples = int(len(self.samples) * self.sample_ratio)
            num_samples = max(1, num_samples)
            self.samples = self.samples[:num_samples]
            print(f"Using {self.sample_ratio*100:.1f}% of {self.split} data: {num_samples} samples")
        
        print(f"Split: {self.split}, Number of samples: {len(self.samples)}")
    
    def _init_mat_format(self):
        """Initialize labeled.mat format dataset (fallback)"""
        self.mat_file = self.data_path / 'raw' / 'nyu_depth_v2_labeled.mat'
        
        if not self.mat_file.exists():
            raise FileNotFoundError(
                f"NYU Depth V2 dataset not found.\n"
                f"Looked for BTS format at: {self.data_path}/nyu/nyu_train.txt\n"
                f"Looked for mat format at: {self.mat_file}\n"
                f"Please download the dataset."
            )
        
        print(f"Loading NYU Depth V2 (mat format) from {self.mat_file}")
        print(f"⚠️  Warning: Using labeled.mat with only 1449 images. "
              f"For better results, use BTS format with 24K images.")
        
        # Store file path for lazy loading
        self.h5_file = None
        self.h5_images = None
        self.h5_depths = None
        
        # Get total samples
        with h5py.File(str(self.mat_file), 'r') as f:
            total_samples = f['images'].shape[0]
        
        # Split: first 1000 for train, last 449 for test
        if self.split == 'train':
            self.start_idx = 0
            self.end_idx = 1000
        else:
            self.start_idx = 1000
            self.end_idx = total_samples
        
        # Apply sample ratio
        num_samples = self.end_idx - self.start_idx
        if self.sample_ratio is not None and 0.0 < self.sample_ratio < 1.0:
            num_samples = int(num_samples * self.sample_ratio)
            num_samples = max(1, num_samples)
            self.end_idx = self.start_idx + num_samples
            print(f"Using {self.sample_ratio*100:.1f}% of {self.split} data: {num_samples} samples")
        
        self.num_samples = self.end_idx - self.start_idx
        print(f"Split: {self.split}, Number of samples: {self.num_samples}")
    
    def __len__(self):
        if self.use_bts_format:
            return len(self.samples)
        else:
            return self.num_samples
    
    def _load_bts_sample(self, idx):
        """Load sample in BTS format"""
        rgb_path, depth_path = self.samples[idx]
        
        # Load RGB image
        rgb_full_path = self.bts_root / rgb_path
        image = Image.open(rgb_full_path).convert('RGB')
        
        # Load depth map (16-bit PNG, values in mm)
        depth_full_path = self.bts_root / depth_path
        depth = Image.open(depth_full_path)
        depth = np.array(depth, dtype=np.float32) / 1000.0  # Convert mm to meters
        depth = Image.fromarray(depth)
        
        return image, depth
    
    def _load_mat_sample(self, idx):
        """Load sample in mat format"""
        # Ensure h5py file is open
        if self.h5_file is None:
            self.h5_file = h5py.File(str(self.mat_file), 'r')
            self.h5_images = self.h5_file['images']
            self.h5_depths = self.h5_file['depths']
        
        actual_idx = self.start_idx + idx
        
        # Load from h5py
        image = self.h5_images[actual_idx]  # (3, 640, 480)
        depth = self.h5_depths[actual_idx]   # (640, 480)
        
        # Convert to (H, W, C) format
        image = np.transpose(image, (2, 1, 0))  # -> (480, 640, 3)
        depth = np.transpose(depth, (1, 0))     # -> (480, 640)
        
        image = Image.fromarray(image.astype(np.uint8))
        depth = Image.fromarray(depth.astype(np.float32))
        
        return image, depth
    
    def __getitem__(self, idx):
        # Load sample based on format
        if self.use_bts_format:
            image, depth = self._load_bts_sample(idx)
        else:
            image, depth = self._load_mat_sample(idx)
        
        # Apply transforms
        if self.augmentation:
            # Random horizontal flip (synchronized)
            if np.random.random() > 0.5:
                image = transforms.functional.hflip(image)
                depth = transforms.functional.hflip(depth)
            
            # Apply image-specific transforms (color jitter, resize)
            image = self.img_transform(image)
            depth = self.depth_transform(depth)
        else:
            image = self.img_transform(image)
            depth = self.depth_transform(depth)
        
        # Convert to tensor
        image = self.to_tensor(image)  # (3, H, W), values in [0, 1]
        image = self.normalize(image)  # Normalized
        
        depth = torch.from_numpy(np.array(depth)).float()  # (H, W)
        
        # Clip depth values
        depth = torch.clamp(depth, self.min_depth, self.max_depth)
        
        return image, depth


def test_dataset():
    """Test the dataset loader"""
    print("=" * 60)
    print("Testing NYU Depth V2 Dataset Loader")
    print("=" * 60)
    
    # Test BTS format
    bts_path = '/home/user/zhoutianjian/DataSets/NYU_BTS'
    try:
        dataset = NYUDepthV2Dataset(
            data_path=bts_path,
            split='train',
            input_size=518
        )
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Format: {'BTS' if dataset.use_bts_format else 'MAT'}")
        print(f"  Size: {len(dataset)}")
        
        # Test loading a sample
        image, depth = dataset[0]
        print(f"  Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


if __name__ == '__main__':
    test_dataset()

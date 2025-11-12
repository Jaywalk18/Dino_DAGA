"""
NYU Depth V2 Dataset Loader
Official dataset: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
Contains 1449 labeled RGB-D image pairs
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
    NYU Depth V2 Dataset
    
    Dataset structure:
    - Images: 1449 RGB images (480x640)
    - Depths: 1449 depth maps (480x640)
    - Train/Test split: Usually first 1000 for train, last 449 for test
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
            data_path: Path to NYUDepthV2 directory
            split: 'train' or 'test'
            input_size: Target image size (default: 518 for DINOv3 depth)
            min_depth: Minimum depth value
            max_depth: Maximum depth value
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
        
        # Path to the labeled .mat file
        self.mat_file = self.data_path / 'raw' / 'nyu_depth_v2_labeled.mat'
        
        if not self.mat_file.exists():
            raise FileNotFoundError(f"NYU Depth V2 mat file not found at {self.mat_file}")
        
        # Load data
        print(f"Loading NYU Depth V2 dataset from {self.mat_file}...")
        print(f"This may take a few minutes for the first time...")
        
        with h5py.File(str(self.mat_file), 'r') as f:
            # Data format in HDF5 file:
            # Images: (1449, 3, 640, 480) - N samples, each is (3, 640, 480)
            # Depths: (1449, 640, 480) - N samples, each is (640, 480)
            images = f['images'][:]  # Shape: (1449, 3, 640, 480)
            depths = f['depths'][:]   # Shape: (1449, 640, 480)
        
        print(f"Loaded images shape: {images.shape}")
        print(f"Loaded depths shape: {depths.shape}")
        
        # Convert to (N, H, W, C) format for images and (N, H, W) for depths
        # Images: (1449, 3, 640, 480) -> (1449, 480, 640, 3)
        # Depths: (1449, 640, 480) -> (1449, 480, 640)
        self.images = np.transpose(images, (0, 3, 2, 1))  # (N, H, W, C)
        self.depths = np.transpose(depths, (0, 2, 1))     # (N, H, W)
        
        print(f"Processed images shape: {self.images.shape}")
        print(f"Processed depths shape: {self.depths.shape}")
        
        # Split dataset: first 1000 for train, last 449 for test
        if split == 'train':
            self.images = self.images[:1000]
            self.depths = self.depths[:1000]
        else:  # test
            self.images = self.images[1000:]
            self.depths = self.depths[1000:]
        
        # Apply sample ratio if specified
        if sample_ratio is not None and 0.0 < sample_ratio < 1.0:
            num_samples = int(len(self.images) * sample_ratio)
            num_samples = max(1, num_samples)  # At least 1 sample
            self.images = self.images[:num_samples]
            self.depths = self.depths[:num_samples]
            print(f"Using {sample_ratio*100:.1f}% of {split} data: {num_samples} samples")
        
        print(f"Split: {split}, Number of samples: {len(self.images)}")
        
        # Define transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Image transforms
        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = transforms.Resize((input_size, input_size))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and depth
        image = self.images[idx]  # (H, W, C), values in [0, 255]
        depth = self.depths[idx]  # (H, W), values in meters
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image.astype(np.uint8))
        depth = Image.fromarray(depth.astype(np.float32))
        
        # Apply same random transforms to both image and depth
        if self.augmentation:
            # Store random state for synchronized transforms
            seed = np.random.randint(2147483647)
            
            # Transform image
            torch.manual_seed(seed)
            image = self.transform(image)
            
            # Transform depth (without color jitter)
            torch.manual_seed(seed)
            depth = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])(depth)
        else:
            image = self.transform(image)
            depth = transforms.Resize((self.input_size, self.input_size))(depth)
        
        # Convert to tensor
        image = self.to_tensor(image)  # (3, H, W), values in [0, 1]
        image = self.normalize(image)  # Normalized
        
        depth = torch.from_numpy(np.array(depth)).float()  # (H, W)
        
        # Clip depth values
        depth = torch.clamp(depth, self.min_depth, self.max_depth)
        
        return image, depth


def test_dataset():
    """Test the dataset loader"""
    dataset = NYUDepthV2Dataset(
        data_path='/home/user/zhoutianjian/DataSets/NYUDepthV2',
        split='train',
        input_size=518
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    image, depth = dataset[0]
    print(f"Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
    
    # Test loading multiple samples
    for i in range(3):
        img, dep = dataset[i]
        print(f"Sample {i}: image {img.shape}, depth {dep.shape}")


if __name__ == '__main__':
    test_dataset()


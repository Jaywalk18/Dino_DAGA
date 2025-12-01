"""
Semantic Segmentation Datasets: ADE20K, VOC2012, and Cityscapes
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ADE20K (150 classes)
ADE20K_NUM_CLASSES = 150

# VOC2012 class names and colors
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
VOC_NUM_CLASSES = 21

# Cityscapes class names (19 classes for evaluation)
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
CITYSCAPES_NUM_CLASSES = 19

# Cityscapes label mapping (from trainId to class index)
# Original labels -> trainId mapping
CITYSCAPES_LABEL_MAP = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,   # road
    8: 1,   # sidewalk
    9: 255, 10: 255,
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 255, 15: 255, 16: 255,
    17: 5,  # pole
    18: 255,
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    29: 255, 30: 255,
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
    -1: 255
}


class VOC2012SegmentationDataset(Dataset):
    """PASCAL VOC 2012 Semantic Segmentation Dataset"""
    
    def __init__(self, root, split='train', input_size=512, transform=None):
        """
        Args:
            root: Path to VOCdevkit/VOC2012 directory
            split: 'train', 'val', or 'trainval'
            input_size: Image size for resizing
            transform: Optional additional transforms
        """
        self.root = root
        self.split = split
        self.input_size = input_size
        self.transform = transform
        
        # Paths
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.mask_dir = os.path.join(root, 'SegmentationClass')
        
        # Load split file
        split_file = os.path.join(root, 'ImageSets', 'Segmentation', f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]
        
        # Filter out images without segmentation masks
        valid_ids = []
        for img_id in self.ids:
            mask_path = os.path.join(self.mask_dir, f'{img_id}.png')
            if os.path.exists(mask_path):
                valid_ids.append(img_id)
        self.ids = valid_ids
        
        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ VOC2012 {split} dataset loaded: {len(self.ids)} samples")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, f'{img_id}.png')
        mask = Image.open(mask_path)
        
        # Resize
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
        
        # Convert mask to tensor
        mask = np.array(mask, dtype=np.int64)
        # VOC uses 255 for boundary/ignore
        mask[mask == 255] = 255  # Keep ignore label
        
        # Apply transforms
        image = self.img_transform(image)
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class CityscapesDataset(Dataset):
    """Cityscapes Semantic Segmentation Dataset"""
    
    def __init__(self, root, split='train', input_size=512, transform=None):
        """
        Args:
            root: Path to Cityscapes directory (containing leftImg8bit and gtFine)
            split: 'train', 'val', or 'test'
            input_size: Image size for resizing
            transform: Optional additional transforms
        """
        self.root = root
        self.split = split
        self.input_size = input_size
        self.transform = transform
        
        # Paths
        self.image_dir = os.path.join(root, 'leftImg8bit', split)
        self.mask_dir = os.path.join(root, 'gtFine', split)
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Collect all images
        self.images = []
        self.masks = []
        
        for city in os.listdir(self.image_dir):
            city_img_dir = os.path.join(self.image_dir, city)
            city_mask_dir = os.path.join(self.mask_dir, city)
            
            if not os.path.isdir(city_img_dir):
                continue
            
            for img_name in os.listdir(city_img_dir):
                if img_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(city_img_dir, img_name)
                    # Corresponding mask
                    mask_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    mask_path = os.path.join(city_mask_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)
        
        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create label mapping array for fast conversion
        self.label_map = np.ones(256, dtype=np.int64) * 255
        for k, v in CITYSCAPES_LABEL_MAP.items():
            if k >= 0:
                self.label_map[k] = v
        
        print(f"✓ Cityscapes {split} dataset loaded: {len(self.images)} samples")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        
        # Load mask
        mask = Image.open(self.masks[idx])
        
        # Resize
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
        
        # Convert mask to numpy and map labels
        mask = np.array(mask, dtype=np.int64)
        mask = self.label_map[mask]
        
        # Apply transforms
        image = self.img_transform(image)
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class ADE20KDataset(Dataset):
    """ADE20K Semantic Segmentation Dataset (150 classes)"""
    
    def __init__(self, root, split='train', input_size=518, transform=None):
        """
        Args:
            root: Path to ADE20K directory (containing images/ADE/training, images/ADE/validation)
            split: 'train' or 'val'
            input_size: Image size for resizing
            transform: Optional additional transforms
        """
        self.root = root
        self.split = split
        self.input_size = input_size
        self.transform = transform
        
        # Map split names
        split_map = {'train': 'training', 'val': 'validation'}
        split_dir = split_map.get(split, split)
        
        # Paths - ADE20K has nested structure: images/ADE/training/category/scene/
        self.base_dir = os.path.join(root, 'images', 'ADE', split_dir)
        
        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"Image directory not found: {self.base_dir}")
        
        # Collect all images recursively
        self.images = []
        self.masks = []
        
        for dirpath, dirnames, filenames in os.walk(self.base_dir):
            for img_name in filenames:
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(dirpath, img_name)
                    # Corresponding mask (same name but _seg.png)
                    mask_name = img_name.replace('.jpg', '_seg.png')
                    mask_path = os.path.join(dirpath, mask_name)
                    
                    if os.path.exists(mask_path):
                        self.images.append(img_path)
                        self.masks.append(mask_path)
        
        # Sort for reproducibility
        sorted_pairs = sorted(zip(self.images, self.masks))
        self.images = [p[0] for p in sorted_pairs]
        self.masks = [p[1] for p in sorted_pairs]
        
        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ ADE20K {split} dataset loaded: {len(self.images)} samples")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.images[idx]).convert('RGB')
        
        # Load mask
        mask = Image.open(self.masks[idx])
        
        # Resize
        image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        mask = mask.resize((self.input_size, self.input_size), Image.NEAREST)
        
        # Convert mask to numpy
        # ADE20K masks: R channel contains class index (0 = background, 1-150 = classes)
        mask = np.array(mask)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take R channel
        mask = mask.astype(np.int64)
        # Shift labels: 0 -> 255 (ignore), 1-150 -> 0-149
        mask = mask - 1
        mask[mask == -1] = 255  # Background becomes ignore
        
        # Apply transforms
        image = self.img_transform(image)
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def get_segmentation_dataset(dataset_name, data_path, split, input_size=512):
    """
    Get segmentation dataset by name
    
    Args:
        dataset_name: 'ade20k', 'voc2012', or 'cityscapes'
        data_path: Root path to dataset
        split: 'train', 'val', or 'test'
        input_size: Image size
    
    Returns:
        Dataset object and number of classes
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'ade20k':
        dataset = ADE20KDataset(data_path, split, input_size)
        num_classes = ADE20K_NUM_CLASSES
    elif dataset_name == 'voc2012':
        dataset = VOC2012SegmentationDataset(data_path, split, input_size)
        num_classes = VOC_NUM_CLASSES
    elif dataset_name == 'cityscapes':
        dataset = CityscapesDataset(data_path, split, input_size)
        num_classes = CITYSCAPES_NUM_CLASSES
    else:
        raise ValueError(f"Unknown segmentation dataset: {dataset_name}")
    
    return dataset, num_classes


if __name__ == '__main__':
    # Test VOC2012
    print("\n=== Testing VOC2012 ===")
    voc_path = "/home/user/zhoutianjian/DataSets/OpenDataLab___PASCAL_VOC2012/raw/VOCdevkit/VOC2012"
    try:
        voc_train, num_cls = get_segmentation_dataset('voc2012', voc_path, 'train')
        print(f"VOC2012 train: {len(voc_train)} samples, {num_cls} classes")
        img, mask = voc_train[0]
        print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test Cityscapes
    print("\n=== Testing Cityscapes ===")
    cs_path = "/home/user/zhoutianjian/DataSets/OpenDataLab___CityScapes/raw"
    try:
        cs_train, num_cls = get_segmentation_dataset('cityscapes', cs_path, 'train')
        print(f"Cityscapes train: {len(cs_train)} samples, {num_cls} classes")
        img, mask = cs_train[0]
        print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")
    except Exception as e:
        print(f"Error: {e}")


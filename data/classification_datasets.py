import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import numpy as np
import scipy.io as sio
from pathlib import Path
from PIL import Image


def get_classification_dataset(args):
    """
    Load and prepare classification dataset.
    Supported: cifar10, cifar100, imagenet, imagenet100, food101, flowers102, pets, cars, sun397, dtd
    """
    if not args.data_path:
        raise ValueError("`--data_path` must be specified.")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.input_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load dataset based on type
    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, download=True, transform=test_transform
        )
        num_classes = 10
        
    elif args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, download=True, transform=test_transform
        )
        num_classes = 100
        
    elif args.dataset == "imagenet100":
        train_path = os.path.join(args.data_path, "train")
        val_path = os.path.join(args.data_path, "val")
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(val_path, transform=test_transform)
        num_classes = 100
        
    elif args.dataset == "imagenet":
        train_path = os.path.join(args.data_path, "train")
        val_path = os.path.join(args.data_path, "val")
        val_annot_path = os.path.join(args.data_path, "val_annotations.txt")
        
        if not os.path.exists(val_annot_path):
            raise FileNotFoundError(f"Annotation file not found: {val_annot_path}")

        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        test_dataset = ImageNetValDataset(
            val_path, val_annot_path, transform=test_transform
        )
        num_classes = len(train_dataset.classes)
    
    elif args.dataset == "food101":
        train_dataset = Food101Dataset(
            root=args.data_path, split="train", transform=train_transform
        )
        test_dataset = Food101Dataset(
            root=args.data_path, split="test", transform=test_transform
        )
        num_classes = 101
    
    elif args.dataset == "flowers102":
        train_dataset = Flowers102Dataset(
            root=args.data_path, split="train", transform=train_transform
        )
        test_dataset = Flowers102Dataset(
            root=args.data_path, split="test", transform=test_transform
        )
        num_classes = 102
    
    elif args.dataset == "pets":
        train_dataset = OxfordPetsDataset(
            root=args.data_path, split="trainval", transform=train_transform
        )
        test_dataset = OxfordPetsDataset(
            root=args.data_path, split="test", transform=test_transform
        )
        num_classes = 37
    
    elif args.dataset == "cars":
        train_dataset = StanfordCarsDataset(
            root=args.data_path, split="train", transform=train_transform
        )
        test_dataset = StanfordCarsDataset(
            root=args.data_path, split="test", transform=test_transform
        )
        num_classes = 196
    
    elif args.dataset == "sun397":
        train_dataset = SUN397Dataset(
            root=args.data_path, split="train", transform=train_transform
        )
        test_dataset = SUN397Dataset(
            root=args.data_path, split="test", transform=test_transform
        )
        num_classes = 397
    
    elif args.dataset == "dtd":
        train_dataset = DTDDataset(
            root=args.data_path, split="train", transform=train_transform
        )
        test_dataset = DTDDataset(
            root=args.data_path, split="test", transform=test_transform
        )
        num_classes = 47
        
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Apply subset if specified
    if hasattr(args, 'subset_ratio') and args.subset_ratio < 1.0:
        subset_size = int(len(train_dataset) * args.subset_ratio)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Using subset of {subset_size} training samples ({args.subset_ratio*100:.1f}%)")

    return train_dataset, test_dataset, num_classes


class ImageNetValDataset(torch.utils.data.Dataset):
    """Custom dataset for ImageNet validation set with annotation file"""
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.targets = []

        # Create a mapping from class ID to index
        train_path = os.path.join(Path(root_dir).parent, "train")
        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(sorted(os.listdir(train_path)))
        }
        self.classes = list(self.class_to_idx.keys())

        # Load annotations
        with open(annotation_file, "r") as f:
            for line in f:
                img_name, class_id = line.strip().split("\t")
                # Support both flat and organized directory structures
                img_path = os.path.join(self.root_dir, img_name)
                if not os.path.exists(img_path):
                    # Try class folder structure
                    img_path = os.path.join(self.root_dir, class_id, img_name)

                if class_id in self.class_to_idx:
                    class_idx = self.class_to_idx[class_id]
                    self.samples.append((img_path, class_idx))
                    self.targets.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = datasets.folder.default_loader(img_path)

        if self.transform:
            image = self.transform(image)

        return image, target


class Food101Dataset(torch.utils.data.Dataset):
    """
    Food-101 Dataset
    
    Structure:
        food-101/
        ├── images/
        │   ├── apple_pie/
        │   │   ├── 1005649.jpg
        │   │   └── ...
        │   └── ...
        └── meta/
            ├── train.txt
            ├── test.txt
            └── classes.txt
    """
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Load class names from classes.txt
        classes_file = self.root / "meta" / "classes.txt"
        if classes_file.exists():
            with open(classes_file, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            # Fallback: get classes from images directory
            self.classes = sorted([d.name for d in (self.root / "images").iterdir() if d.is_dir()])
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load split file (train.txt or test.txt)
        split_file = self.root / "meta" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, "r") as f:
            for line in f:
                # Format: class_name/image_id (without extension)
                relative_path = line.strip()
                class_name = relative_path.split("/")[0]
                
                # Construct full image path
                img_path = self.root / "images" / f"{relative_path}.jpg"
                
                if class_name in self.class_to_idx:
                    class_idx = self.class_to_idx[class_name]
                    self.samples.append((str(img_path), class_idx))
                    self.targets.append(class_idx)
        
        print(f"Food-101 {split}: loaded {len(self.samples)} samples, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


class Flowers102Dataset(torch.utils.data.Dataset):
    """Oxford 102 Flowers Dataset (102 classes)"""
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Load labels and split info from .mat files
        labels_mat = sio.loadmat(self.root / "imagelabels.mat")
        setid_mat = sio.loadmat(self.root / "setid.mat")
        
        all_labels = labels_mat['labels'][0]  # 1-indexed labels
        
        # Get split indices (1-indexed in mat file)
        if split == "train":
            indices = setid_mat['trnid'][0]
        elif split == "val":
            indices = setid_mat['valid'][0]
        else:  # test
            indices = setid_mat['tstid'][0]
        
        # Build samples list
        img_dir = self.root / "jpg"
        for idx in indices:
            img_path = img_dir / f"image_{idx:05d}.jpg"
            label = all_labels[idx - 1] - 1  # Convert to 0-indexed
            self.samples.append((str(img_path), label))
            self.targets.append(label)
        
        self.classes = list(range(102))
        print(f"Flowers-102 {split}: loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


class OxfordPetsDataset(torch.utils.data.Dataset):
    """Oxford-IIIT Pets Dataset (37 classes)"""
    def __init__(self, root, split="trainval", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Load split file
        split_file = self.root / "annotations" / f"{split}.txt"
        img_dir = self.root / "images"
        
        # Build class mapping from image names (class is prefix before last _)
        class_names = set()
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                # Class name is everything before the last underscore and number
                class_name = "_".join(img_name.split("_")[:-1])
                class_names.add(class_name)
        
        self.classes = sorted(list(class_names))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Load samples
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                class_name = "_".join(img_name.split("_")[:-1])
                
                img_path = img_dir / f"{img_name}.jpg"
                if img_path.exists():
                    label = self.class_to_idx[class_name]
                    self.samples.append((str(img_path), label))
                    self.targets.append(label)
        
        print(f"Oxford Pets {split}: loaded {len(self.samples)} samples, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


class StanfordCarsDataset(torch.utils.data.Dataset):
    """Stanford Cars Dataset (196 classes)
    Note: Test set has no labels, so we split train set into train/val
    """
    def __init__(self, root, split="train", transform=None, train_ratio=0.8):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Load train annotations (test set has no labels)
        annos_mat = sio.loadmat(self.root / "devkit" / "cars_train_annos.mat")
        img_dir = self.root / "cars_train"
        annotations = annos_mat['annotations'][0]
        
        all_samples = []
        for anno in annotations:
            # anno: (bbox_x1, bbox_y1, bbox_x2, bbox_y2, class, fname)
            img_name = str(anno[5][0])
            label = int(anno[4][0][0]) - 1  # Convert to 0-indexed
            
            img_path = img_dir / img_name
            if img_path.exists():
                all_samples.append((str(img_path), label))
        
        # Split train/test from train set (test set has no labels)
        np.random.seed(42)
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        self.targets = [s[1] for s in self.samples]
        self.classes = list(range(196))
        print(f"Stanford Cars {split}: loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


class SUN397Dataset(torch.utils.data.Dataset):
    """SUN397 Scene Dataset (397 classes)"""
    def __init__(self, root, split="train", transform=None, train_ratio=0.8):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Load class names
        class_file = self.root / "ClassName.txt"
        with open(class_file, "r") as f:
            self.classes = [line.strip().lstrip('/') for line in f.readlines()]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Collect all images
        all_samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    label = self.class_to_idx[class_name]
                    all_samples.append((str(img_path), label))
        
        # Split train/test (no official split, use random)
        np.random.seed(42)
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        self.targets = [s[1] for s in self.samples]
        print(f"SUN397 {split}: loaded {len(self.samples)} samples, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


class DTDDataset(torch.utils.data.Dataset):
    """Describable Textures Dataset (47 classes)"""
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Use split1 by default
        labels_dir = self.root / "labels"
        images_dir = self.root / "images"
        
        # Load split file
        split_file = labels_dir / f"{split}1.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"DTD split file not found: {split_file}")
        
        # Build class list from image paths
        class_names = set()
        with open(split_file, "r") as f:
            for line in f:
                # Format: class_name/image.jpg
                rel_path = line.strip()
                class_name = rel_path.split("/")[0]
                class_names.add(class_name)
        
        self.classes = sorted(list(class_names))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Load samples
        with open(split_file, "r") as f:
            for line in f:
                rel_path = line.strip()
                class_name = rel_path.split("/")[0]
                img_path = images_dir / rel_path
                
                if img_path.exists():
                    label = self.class_to_idx[class_name]
                    self.samples.append((str(img_path), label))
                    self.targets.append(label)
        
        print(f"DTD {split}: loaded {len(self.samples)} samples, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target

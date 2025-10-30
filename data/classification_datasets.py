import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from pathlib import Path


def get_classification_dataset(args):
    """
    Load and prepare classification dataset (CIFAR10, CIFAR100, ImageNet)
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
                img_path = os.path.join(self.root_dir, img_name)

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

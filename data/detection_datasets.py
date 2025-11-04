import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset
import os


def get_detection_dataset(args):
    """
    Load and prepare detection dataset (COCO)
    """
    if args.dataset == "coco":
        return get_coco_dataset(args)
    else:
        raise ValueError(f"Unknown detection dataset: {args.dataset}")


def get_coco_dataset(args):
    """Load COCO 2017 dataset for object detection"""
    from pycocotools.coco import COCO
    
    # COCO paths
    train_img_dir = os.path.join(args.data_path, "train2017")
    val_img_dir = os.path.join(args.data_path, "val2017")
    train_ann_file = os.path.join(args.data_path, "annotations/instances_train2017.json")
    val_ann_file = os.path.join(args.data_path, "annotations/instances_val2017.json")
    
    # Verify paths exist
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"COCO train images not found: {train_img_dir}")
    if not os.path.exists(val_img_dir):
        raise FileNotFoundError(f"COCO val images not found: {val_img_dir}")
    if not os.path.exists(train_ann_file):
        raise FileNotFoundError(f"COCO train annotations not found: {train_ann_file}")
    if not os.path.exists(val_ann_file):
        raise FileNotFoundError(f"COCO val annotations not found: {val_ann_file}")
    
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
    
    # Load datasets
    train_dataset = COCODetectionDataset(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        transform=train_transform,
        input_size=args.input_size
    )
    
    val_dataset = COCODetectionDataset(
        img_dir=val_img_dir,
        ann_file=val_ann_file,
        transform=val_transform,
        input_size=args.input_size
    )
    
    # COCO has 80 object categories
    num_classes = 80
    
    return train_dataset, val_dataset, num_classes


class COCODetectionDataset(Dataset):
    """COCO Detection Dataset"""
    def __init__(self, img_dir, ann_file, transform=None, input_size=518):
        from pycocotools.coco import COCO
        
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.input_size = input_size
        
        # Filter out images without annotations
        self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        
        print(f"Found {len(self.ids)} images with annotations")
        
        # Validate a few samples
        if len(self.ids) > 0:
            self._validate_samples(min(3, len(self.ids)))
    
    def _validate_samples(self, num_samples=3):
        """Validate a few samples to check bbox ranges"""
        print(f"Validating {num_samples} detection samples...")
        for i in range(num_samples):
            img_id = self.ids[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            num_boxes = len([ann for ann in anns if 'bbox' in ann and ann['area'] > 0])
            print(f"  Sample {i} (img_id={img_id}): {num_boxes} valid boxes")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        from PIL import Image
        
        img_id = self.ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get original size
        orig_w, orig_h = image.size
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        
        for ann in anns:
            if 'bbox' in ann and ann['area'] > 0:
                x, y, w, h = ann['bbox']
                # Convert to [x1, y1, x2, y2] format
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            # Scale boxes to match resized image
            scale_x = self.input_size / orig_w
            scale_y = self.input_size / orig_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        return image, target


def detection_collate_fn(batch):
    """Custom collate function for detection datasets"""
    images = []
    boxes_list = []
    labels_list = []
    
    for img, target in batch:
        images.append(img)
        boxes_list.append(target['boxes'])
        labels_list.append(target['labels'])
    
    images = torch.stack(images, dim=0)
    
    return images, boxes_list, labels_list

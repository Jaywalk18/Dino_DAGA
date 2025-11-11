"""
Detection Metrics Test Script
Tests the detection model metrics (mAP, Precision, Recall) using pretrained checkpoints
Purpose: Verify that Precision and Recall are correctly calculated
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.backbones import load_dinov3_backbone
from tasks.detection import DetectionModel, detection_collate_fn
from data.detection_datasets import get_detection_dataset
from core.simple_detection_head import decode_predictions, box_iou


def compute_detailed_map_metrics(predictions, gt_boxes_list, gt_labels_list, num_classes, iou_threshold=0.5):
    """
    Compute mAP and detailed metrics (Precision, Recall, F1)
    Returns both per-class and overall metrics
    """
    # Organize ground truth by class
    cls_gts = {}  # cls -> [list of (img_id, gt_boxes)]
    total_gts_per_class = {}
    
    for img_id, (gt_boxes, gt_labels) in enumerate(zip(gt_boxes_list, gt_labels_list)):
        for box, label in zip(gt_boxes, gt_labels):
            cls = label.item()
            if cls not in cls_gts:
                cls_gts[cls] = {}
                total_gts_per_class[cls] = 0
            if img_id not in cls_gts[cls]:
                cls_gts[cls][img_id] = []
            cls_gts[cls][img_id].append(box)
            total_gts_per_class[cls] += 1
    
    # Organize predictions by class
    cls_preds = {}  # cls -> list of (img_id, box, score)
    for img_id, pred in enumerate(predictions):
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']
        
        for box, score, label in zip(boxes, scores, labels):
            cls = label.item()
            if cls not in cls_preds:
                cls_preds[cls] = []
            cls_preds[cls].append((img_id, box, score))
    
    # Compute AP for each class and collect overall TP/FP
    aps = []
    all_tp = 0
    all_fp = 0
    all_gt = sum(total_gts_per_class.values())
    
    for cls in range(num_classes):
        if cls not in cls_gts or cls not in cls_preds:
            if cls in total_gts_per_class:
                aps.append(0.0)
            continue
        
        # Get predictions for this class
        preds_cls = cls_preds[cls]
        preds_cls = sorted(preds_cls, key=lambda x: x[2], reverse=True)  # Sort by score
        
        # Compute TP/FP
        tp = []
        fp = []
        matched = {img_id: set() for img_id in cls_gts[cls].keys()}
        
        for img_id, box, score in preds_cls:
            if img_id not in cls_gts[cls]:
                fp.append(1)
                tp.append(0)
                all_fp += 1
                continue
            
            gt_boxes = torch.stack(cls_gts[cls][img_id])
            ious = box_iou(box.unsqueeze(0), gt_boxes)[0]
            max_iou, max_idx = ious.max(dim=-1)
            
            if max_iou >= iou_threshold and max_idx.item() not in matched[img_id]:
                tp.append(1)
                fp.append(0)
                matched[img_id].add(max_idx.item())
                all_tp += 1
            else:
                tp.append(0)
                fp.append(1)
                all_fp += 1
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / total_gts_per_class[cls]
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if mask.sum() > 0:
                ap += precisions[mask].max()
        ap /= 11.0
        
        aps.append(ap)
    
    # Calculate overall metrics
    mAP = sum(aps) / len(aps) if len(aps) > 0 else 0.0
    
    # Overall Precision and Recall
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_recall = all_tp / all_gt if all_gt > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-10)
    
    return {
        'mAP': mAP,
        'mAP@50': mAP,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': all_tp,
        'fp': all_fp,
        'fn': all_gt - all_tp,
        'total_gt': all_gt,
        'num_predictions': all_tp + all_fp,
    }


def evaluate_model(model, dataloader, device, num_classes):
    """Evaluate detection model with detailed metrics"""
    model.eval()
    total_loss = 0.0
    
    all_predictions = []
    all_gt_boxes = []
    all_gt_labels = []
    
    with torch.no_grad():
        for images, boxes_list, labels_list in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            boxes_list = [boxes.to(device) for boxes in boxes_list]
            labels_list = [labels.to(device) for labels in labels_list]
            
            _, H, W = images[0].shape
            
            # Get model predictions
            cls_logits, box_preds, centerness, _, _ = model(images, False)
            
            # Decode predictions
            stride = model.stride if not isinstance(model, nn.DataParallel) else model.module.stride
            detections = decode_predictions(
                cls_logits, box_preds, centerness,
                image_size=(H, W),
                stride=stride,
                score_threshold=0.05,
                nms_threshold=0.5,
                max_detections=100
            )
            
            all_predictions.extend(detections)
            all_gt_boxes.extend(boxes_list)
            all_gt_labels.extend(labels_list)
    
    # Compute detailed metrics
    metrics = compute_detailed_map_metrics(
        all_predictions, all_gt_boxes, all_gt_labels, 
        num_classes, iou_threshold=0.5
    )
    
    return metrics


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    args = argparse.Namespace(**checkpoint['args'])
    
    # Load backbone
    vit_model = load_dinov3_backbone(args.model_name, args.pretrained_path)
    
    # Create detection model
    model = DetectionModel(
        vit_model,
        num_classes=args.num_classes,
        use_daga=args.use_daga,
        daga_layers=args.daga_layers,
        layers_to_use=args.layers_to_use,
        enable_visualization=False,
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  - Use DAGA: {args.use_daga}")
    if args.use_daga:
        print(f"  - DAGA Layers: {args.daga_layers}")
    print(f"  - Layers to use: {args.layers_to_use}")
    
    return model, args


def main():
    parser = argparse.ArgumentParser(description="Test Detection Metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data_path", type=str, default="/home/user/zhoutianjian/DataSets/COCO 2017", 
                       help="Path to COCO dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="Fraction of dataset to use (for quick testing)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Detection Metrics Testing")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*70}\n")
    
    # Load model
    model, model_args = load_model_from_checkpoint(args.checkpoint, device)
    
    # Load dataset
    print("\n📊 Loading COCO validation dataset...")
    _, val_dataset, num_classes = get_detection_dataset(
        dataset_name='coco',
        data_path=args.data_path,
        input_size=model_args.input_size,
        sample_ratio=args.sample_ratio
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True
    )
    
    print(f"✓ Dataset loaded: {len(val_dataset)} validation samples")
    print(f"  Using {args.sample_ratio*100:.1f}% of dataset ({len(val_dataset)} samples)")
    
    # Evaluate
    print("\n🔍 Evaluating model...")
    metrics = evaluate_model(model, val_loader, device, num_classes)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"📈 Detection Metrics Results")
    print(f"{'='*70}")
    print(f"mAP@50:      {metrics['mAP@50']*100:.2f}%")
    print(f"Precision:   {metrics['precision']*100:.2f}%")
    print(f"Recall:      {metrics['recall']*100:.2f}%")
    print(f"F1 Score:    {metrics['f1']*100:.2f}%")
    print(f"")
    print(f"True Positives:   {metrics['tp']}")
    print(f"False Positives:  {metrics['fp']}")
    print(f"False Negatives:  {metrics['fn']}")
    print(f"Total GT Boxes:   {metrics['total_gt']}")
    print(f"Total Predictions: {metrics['num_predictions']}")
    print(f"{'='*70}\n")
    
    # Save results
    checkpoint_dir = Path(args.checkpoint).parent
    results_file = checkpoint_dir / "detailed_metrics.txt"
    
    with open(results_file, "w") as f:
        f.write("Detection Metrics Test Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: COCO validation ({args.sample_ratio*100:.1f}% sampled)\n")
        f.write(f"Samples evaluated: {len(val_dataset)}\n\n")
        f.write("="*50 + "\n")
        f.write("Metrics:\n")
        f.write("="*50 + "\n")
        f.write(f"mAP@50:      {metrics['mAP@50']*100:.2f}%\n")
        f.write(f"Precision:   {metrics['precision']*100:.2f}%\n")
        f.write(f"Recall:      {metrics['recall']*100:.2f}%\n")
        f.write(f"F1 Score:    {metrics['f1']*100:.2f}%\n\n")
        f.write("Detailed Statistics:\n")
        f.write(f"True Positives:   {metrics['tp']}\n")
        f.write(f"False Positives:  {metrics['fp']}\n")
        f.write(f"False Negatives:  {metrics['fn']}\n")
        f.write(f"Total GT Boxes:   {metrics['total_gt']}\n")
        f.write(f"Total Predictions: {metrics['num_predictions']}\n")
    
    print(f"✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()


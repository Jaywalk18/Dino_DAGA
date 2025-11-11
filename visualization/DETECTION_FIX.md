# Detection Metrics Fix

## 问题
在 `tasks/detection.py` 的 `evaluate` 函数中，Precision 和 Recall 被硬编码为 0.0：

```python
# 第607-617行 (旧代码 - 有问题)
return {
    'loss': avg_loss, 
    'mAP': mAP, 
    'mAP@50': mAP,
    'precision': 0.0,   # ❌ 硬编码
    'recall': 0.0,      # ❌ 硬编码
    'f1': 0.0, 
    'tp': 0, 
    'fp': 0, 
    'total_gt': 0
}
```

## 解决方案

### 步骤1: 修改 `compute_map` 函数
让它返回更详细的指标，包括 overall TP, FP, 和总 GT数量。

在 `tasks/detection.py` 中，将 `compute_map` 函数修改为：

```python
def compute_map(predictions, gt_boxes_list, gt_labels_list, num_classes, iou_threshold=0.5):
    """
    Compute mean Average Precision and detailed metrics
    Returns: tuple of (mAP, overall_tp, overall_fp, total_gt)
    """
    # ... 现有代码保持不变 ...
    
    # 在循环中累计所有类别的 TP/FP
    aps = []
    all_tp = 0      # 添加这行
    all_fp = 0      # 添加这行
    all_gt = 0      # 添加这行
    
    for cls in range(num_classes):
        if cls not in cls_gts or cls not in cls_preds:
            if cls in cls_gts:
                aps.append(0.0)
                all_gt += sum(len(boxes) for boxes in cls_gts[cls].values())  # 添加这行
            continue
        
        # 计算这个类别的total GT
        total_gt = sum(len(boxes) for boxes in cls_gts[cls].values())
        all_gt += total_gt  # 添加这行
        
        # ... 现有的 TP/FP 计算代码 ...
        
        # 在循环中累计 TP/FP
        for i, (box, img_id) in enumerate(zip(all_boxes, all_img_ids)):
            # ... 现有代码 ...
            if max_iou >= iou_threshold and max_idx.item() not in matched[img_id]:
                tp[i] = 1
                all_tp += 1      # 添加这行
                matched[img_id].add(max_idx.item())
            else:
                fp[i] = 1
                all_fp += 1      # 添加这行
        
        # ... AP 计算保持不变 ...
    
    # 返回 mAP 和统计信息
    mAP = sum(aps) / len(aps) if len(aps) > 0 else 0.0
    return mAP, all_tp, all_fp, all_gt
```

### 步骤2: 修改 `evaluate` 函数
使用 `compute_map` 返回的详细指标。

将第605-617行替换为：

```python
# Compute mAP and detailed metrics
mAP, total_tp, total_fp, total_gt = compute_map(
    all_predictions, all_gt_boxes, all_gt_labels, num_classes, iou_threshold=0.5
)

# Calculate precision and recall
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
recall = total_tp / total_gt if total_gt > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

return {
    'loss': avg_loss, 
    'mAP': mAP, 
    'mAP@50': mAP,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'tp': total_tp,
    'fp': total_fp,
    'fn': total_gt - total_tp,
    'total_gt': total_gt
}
```

## 测试步骤

1. 先运行测试脚本验证新逻辑：
   ```bash
   bash visualization/test_detection.sh
   ```

2. 检查输出，确保 Precision 和 Recall 不为 0

3. 如果测试通过，应用上述修改到 `tasks/detection.py`

4. 重新运行完整的detection训练来验证

## 预期结果

修改后，应该看到类似这样的输出：
```
mAP@50:      0.50%
Precision:   12.34%   # 不再是 0.00%
Recall:      8.76%    # 不再是 0.00%
F1 Score:    10.23%
```


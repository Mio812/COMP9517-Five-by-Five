
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def load_names(data_root: Path):
    yaml_path = data_root / 'data.yaml'
    if yaml_path.exists():
        cfg = yaml.safe_load(open(yaml_path))
        return cfg['names']


def yolo_to_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    xc, yc, bw, bh = box
    xc *= w; yc *= h; bw *= w; bh *= h
    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def resize_short_side(image: Image.Image, boxes: torch.Tensor, target_short: int):
    w, h = image.size
    short = min(w, h)
    if short == target_short:
        return image, boxes
    scale = target_short / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BILINEAR)
    if boxes.numel() > 0:
        boxes = boxes * torch.tensor([scale, scale, scale, scale])
    return image, boxes


class SimpleDataset(Dataset):
    def __init__(self, root: Path, split: str, class_names, imgsz: int = 640):
        self.root = Path(root)
        self.split = split
        self.class_names = class_names
        self.img_dir = self.root / self.split / "images"
        self.lbl_dir = self.root / self.split / "labels"
        self.img_paths = sorted([p for p in self.img_dir.rglob("*") 
                                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.imgsz = imgsz

    def __len__(self):
        return len(self.img_paths)

    def _read_labels(self, lbl_path: Path, W: int, H: int):
        boxes, labels = [], []
        if not lbl_path.exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
        lines = [ln.strip() for ln in lbl_path.read_text().strip().splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
            xyxy = yolo_to_xyxy(np.array([xc, yc, bw, bh], dtype=np.float32), W, H)
            x1, y1, x2, y2 = xyxy
            x1 = max(0.0, min(x1, W - 1))
            y1 = max(0.0, min(y1, H - 1))
            x2 = max(0.0, min(x2, W - 1))
            y2 = max(0.0, min(y2, H - 1))
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cid + 1)
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        lbl_path = (self.lbl_dir / img_path.name).with_suffix('.txt')
        image = Image.open(img_path).convert('RGB')
        W, H = image.size
        boxes, labels = self._read_labels(lbl_path, W, H)
        image, boxes = resize_short_side(image, boxes, self.imgsz)
        image_t = to_tensor(image)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
        }
        return image_t, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def load_model(checkpoint_path, num_classes, device):
    """加载训练好的模型"""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes + 1
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def evaluate_metrics(model, dataloader, device, conf_threshold=0.001, iou_threshold=0.5, num_classes=None):
    """评估所有指标"""
    model.eval()
    
    # 用于计算 Precision, Recall, F1
    TP = 0
    FP = 0
    FN = 0
    
    # 用于计算 mAP
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=False)
    
    # 用于计算 Accuracy (混淆矩阵)
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    
    print("Evaluating...")
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            # 处理每个图像
            for out, tgt in zip(outputs, targets):
                # 过滤低置信度预测
                mask = out['scores'] >= conf_threshold
                pred_boxes = out['boxes'][mask].cpu()
                pred_scores = out['scores'][mask].cpu()
                pred_labels = out['labels'][mask].cpu()
                
                gt_boxes = tgt['boxes'].cpu()
                gt_labels = tgt['labels'].cpu()
                
                # 计算 IoU 并匹配
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)
                    
                    # 贪婪匹配
                    matched_pred = set()
                    matched_gt = set()
                    
                    # 按照 IoU 从大到小排序
                    iou_values = ious.numpy()
                    indices = np.dstack(np.unravel_index(np.argsort(-iou_values.ravel()), iou_values.shape))[0]
                    
                    for i, j in indices:
                        if iou_values[i, j] < iou_threshold:
                            break
                        if i in matched_pred or j in matched_gt:
                            continue
                        if pred_labels[i] == gt_labels[j]:
                            matched_pred.add(i)
                            matched_gt.add(j)
                            TP += 1
                            # 更新混淆矩阵（正确预测）
                            confusion_matrix[gt_labels[j], pred_labels[i]] += 1
                    
                    # FP: 未匹配的预测
                    FP += len(pred_boxes) - len(matched_pred)
                    
                    # FN: 未匹配的真实框
                    FN += len(gt_boxes) - len(matched_gt)
                    
                    # 混淆矩阵：未匹配的预测作为误分类
                    for i in range(len(pred_boxes)):
                        if i not in matched_pred:
                            # 背景类（索引 0）
                            confusion_matrix[0, pred_labels[i]] += 1
                    
                    for j in range(len(gt_boxes)):
                        if j not in matched_gt:
                            # 漏检
                            confusion_matrix[gt_labels[j], 0] += 1
                else:
                    FP += len(pred_boxes)
                    FN += len(gt_boxes)
                    for label in pred_labels:
                        confusion_matrix[0, label] += 1
                    for label in gt_labels:
                        confusion_matrix[label, 0] += 1
            
            # 准备 mAP 计算的数据
            outputs_cpu = []
            for o in outputs:
                outputs_cpu.append({
                    'boxes': o['boxes'].detach().cpu(),
                    'scores': o['scores'].detach().cpu(),
                    'labels': o['labels'].detach().cpu(),
                })
            targets_cpu = []
            for t in targets:
                targets_cpu.append({
                    'boxes': t['boxes'].detach().cpu(),
                    'labels': t['labels'].detach().cpu(),
                })
            metric.update(outputs_cpu, targets_cpu)
    
    # 计算 Precision, Recall, F1
    P = TP / (TP + FP + 1e-12)
    R = TP / (TP + FN + 1e-12)
    F1 = 2 * P * R / (P + R + 1e-12)
    
    # 计算 Accuracy (从混淆矩阵)
    # 只考虑非背景类
    acc = np.trace(confusion_matrix[1:, 1:]) / (confusion_matrix[1:, 1:].sum() + 1e-12)
    
    # 计算 mAP
    map_result = metric.compute()
    AUC_05 = float(map_result['map_50'].item()) if 'map_50' in map_result else 0.0
    AUC_5095 = float(map_result['map'].item()) if 'map' in map_result else 0.0
    
    return {
        'Precision': P,
        'Recall': R,
        'F1': F1,
        'Accuracy': acc,
        'AUC_05': AUC_05,
        'AUC_5095': AUC_5095,
        'confusion_matrix': confusion_matrix
    }


def save_plots(results, output_dir):
    """保存可视化图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    P = results['Precision']
    R = results['Recall']
    F1 = results['F1']
    acc = results['Accuracy']
    AUC_05 = results['AUC_05']
    AUC_5095 = results['AUC_5095']
    
    # 1. 柱状图
    plt.figure(figsize=(12, 7))
    metrics = ['Test Precision', 'Test Recall', 'Test F1', 
               'Test Accuracy', 'Test AUC@0.5', 'Test AUC@0.5:0.95']
    values = [P, R, F1, acc, AUC_05, AUC_5095]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylim(0, 1.0)
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.title('Faster R-CNN Test Set Evaluation Metrics', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=30, ha='right', fontsize=11)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{value:.4f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'test_metrics.png'}")
    plt.close()
    
    # 2. 雷达图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]
    
    ax.plot(angles, values_plot, 'o-', linewidth=2.5, color='#1f77b4', markersize=8)
    ax.fill(angles, values_plot, alpha=0.25, color='#1f77b4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Precision', 'Recall', 'F1', 'Accuracy', 'AUC@0.5', 'AUC@0.5:0.95'], 
                       fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title('Test Metrics Radar Chart', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_metrics_radar.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'test_metrics_radar.png'}")
    plt.close()
    
    # 3. 保存文本结果
    with open(output_dir / 'test_results.txt', 'w') as f:
        f.write(f"Test Precision: {P:.4f}\n")
        f.write(f"Test Recall:    {R:.4f}\n")
        f.write(f"Test F1:        {F1:.4f}\n")
        f.write(f"Test Accuracy:  {acc:.4f}\n")
        f.write(f"Test AUC@0.5:   {AUC_05:.4f}\n")
        f.write(f"Test AUC@0.5:0.95: {AUC_5095:.4f}\n")
    
    print(f"✓ Saved: {output_dir / 'test_results.txt'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Faster R-CNN Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/dataset', help='Dataset root')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'valid'], help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf_threshold', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for matching')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/cnn/test_frcnn', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    print("="*70)
    print("Faster R-CNN Model Evaluation")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data_root}/{args.split}")
    print(f"Output: {args.output_dir}")
    print("="*70)
    
    # 加载数据
    data_root = Path(args.data_root)
    class_names = load_names(data_root)
    num_classes = len(class_names)
    
    dataset = SimpleDataset(data_root, args.split, class_names, imgsz=args.imgsz)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    # 加载模型
    device = torch.device(args.device)
    model = load_model(args.model_path, num_classes, device)
    print(f"✓ Model loaded from {args.model_path}")
    
    # 评估
    results = evaluate_metrics(model, dataloader, device, 
                              conf_threshold=args.conf_threshold,
                              iou_threshold=args.iou_threshold,
                              num_classes=num_classes)
    
    # 打印结果
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"Test Precision: {results['Precision']:.4f}")
    print(f"Test Recall:    {results['Recall']:.4f}")
    print(f"Test F1:        {results['F1']:.4f}")
    print(f"Test Accuracy:  {results['Accuracy']:.4f}")
    print(f"Test AUC@0.5:   {results['AUC_05']:.4f}")
    print(f"Test AUC@0.5:0.95: {results['AUC_5095']:.4f}")
    print("="*70)
    
    # 保存图表
    save_plots(results, args.output_dir)
    
    print(f"\n✓ All results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

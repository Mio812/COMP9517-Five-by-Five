import argparse
import os
from pathlib import Path
from typing import List, Dict
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import time
from collections import defaultdict


sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300


def yolo_to_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    xc, yc, bw, bh = box
    x1, y1 = (xc - bw / 2.0) * w, (yc - bh / 2.0) * h
    x2, y2 = (xc + bw / 2.0) * w, (yc + bh / 2.0) * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def load_class_names(data_root: Path) -> List[str]:
    yaml_path = data_root / 'data.yaml'
    if not yaml_path.exists():
        yaml_path = data_root.parent / 'data.yaml'
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {data_root} or its parent.")
    with open(yaml_path, 'r') as f: cfg = yaml.safe_load(f)
    if 'names' not in cfg: raise KeyError("'names' key not found in data.yaml")
    names = cfg['names']
    if isinstance(names, dict): names = [names[i] for i in sorted(names.keys())]
    return names

class TestDataset(Dataset):
    def __init__(self, root: Path, split: str, class_names: List[str], imgsz: int = 640):
        self.root = root
        self.split = split
        self.img_dir = root / split / "images"
        self.lbl_dir = root / split / "labels"
        self.img_paths = sorted([p for p in self.img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.imgsz = imgsz
        self.class_names = class_names
        self.num_classes = len(class_names)
        print(f"Loaded {len(self.img_paths)} images for '{split}' from {self.img_dir}")
    def _read_labels(self, lbl_path: Path, W: int, H: int):
        boxes, labels = [], []
        if not lbl_path.exists(): return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)
        lines = [ln.strip() for ln in lbl_path.read_text().strip().splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5: continue
            cid = int(float(parts[0]))
            if not (0 <= cid < self.num_classes): continue
            xyxy = yolo_to_xyxy(np.array(list(map(float, parts[1:5]))), W, H)
            if (xyxy[2] - xyxy[0]) > 1 and (xyxy[3] - xyxy[1]) > 1:
                boxes.append(xyxy.tolist()); labels.append(cid)
        if not boxes: return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB'); W, H = image.size
        lbl_path = (self.lbl_dir / img_path.name).with_suffix('.txt')
        boxes, labels = self._read_labels(lbl_path, W, H)
        scale_w, scale_h = self.imgsz / W, self.imgsz / H
        image = image.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale_w; boxes[:, [1, 3]] *= scale_h
            boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], 0, self.imgsz)
            boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], 0, self.imgsz)
        return to_tensor(image), boxes, labels, str(img_path)

def collate_fn(batch):
    images, boxes, labels, paths = zip(*batch)
    return list(images), list(boxes), list(labels), list(paths)

class FasterRCNNDetector(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    def forward(self, images, targets=None):
        return self.model(images, targets)

@torch.no_grad()
def evaluate_model(model, dataloader, device, conf_thres=0.5):
    model.eval()
    all_pred_boxes, all_pred_labels, all_pred_scores = [], [], []
    all_gt_boxes, all_gt_labels = [], []
    inference_times = []
    print(f"\nEvaluating on {len(dataloader.dataset)} test images...")
    for images, boxes, labels, paths in tqdm(dataloader, desc="Testing"):
        images = [img.to(device) for img in images]
        start_time = time.time()
        predictions = model(images)
        inference_times.append((time.time() - start_time) / len(images))
        for pred, gt_box, gt_label in zip(predictions, boxes, labels):
            mask = pred['scores'] > conf_thres
            all_pred_boxes.append(pred['boxes'][mask].cpu())
            all_pred_labels.append((pred['labels'][mask] - 1).cpu())
            all_pred_scores.append(pred['scores'][mask].cpu())
            all_gt_boxes.append(gt_box)
            all_gt_labels.append(gt_label)
    avg_inference_time = np.mean(inference_times)
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms per image")
    return (all_pred_boxes, all_pred_labels, all_pred_scores, all_gt_boxes, all_gt_labels, avg_inference_time)

def calculate_metrics(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, num_classes, class_names, avg_inference_time):
    preds, targets = [], []
    for pb, pl, ps, gb, gl in zip(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        preds.append({'boxes': pb, 'labels': pl, 'scores': ps})
        targets.append({'boxes': gb, 'labels': gl})

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox'); metric.update(preds, targets)
    map_metrics = metric.compute()
    matched_preds, matched_targets, matched_scores = [], [], []
    for pb, pl, ps, gb, gl in zip(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        if len(gl) == 0: continue
        if len(pl) == 0:
            matched_targets.extend(gl.numpy().tolist()); matched_preds.extend([-1] * len(gl))
            matched_scores.extend([np.zeros(num_classes)] * len(gl)); continue
        iou_matrix = box_iou_numpy(pb.numpy(), gb.numpy()); matched_gt = set()
        for i in range(len(pb)):
            if len(gb) == 0: continue
            best_iou_idx = iou_matrix[i].argmax(); best_iou = iou_matrix[i][best_iou_idx]
            if best_iou > 0.5 and best_iou_idx not in matched_gt:
                matched_gt.add(best_iou_idx)
                matched_targets.append(gl[best_iou_idx].item()); matched_preds.append(pl[i].item())
                pred_label, pred_score = pl[i].item(), ps[i].item(); prob_vector = np.zeros(num_classes)
                prob_vector[pred_label] = pred_score
                if num_classes > 1:
                    remaining_prob = (1.0 - pred_score) / (num_classes - 1)
                    prob_vector += remaining_prob; prob_vector[pred_label] = pred_score
                matched_scores.append(prob_vector)
        for j in range(len(gl)):
            if j not in matched_gt:
                matched_targets.append(gl[j].item()); matched_preds.append(-1)
                matched_scores.append(np.zeros(num_classes))
    auc = 0.0; valid_mask = np.array(matched_preds) != -1
    if valid_mask.sum() > 0:
        valid_preds = np.array(matched_preds)[valid_mask]; valid_targets = np.array(matched_targets)[valid_mask]
        valid_scores = np.array(matched_scores)[valid_mask]
        precision = precision_score(valid_targets, valid_preds, average='weighted', zero_division=0)
        f1 = f1_score(valid_targets, valid_preds, average='weighted', zero_division=0)
        accuracy = accuracy_score(valid_targets, valid_preds)
        if len(np.unique(valid_targets)) > 1:
            y_true_binarized = label_binarize(valid_targets, classes=range(num_classes))
            auc = roc_auc_score(y_true_binarized, valid_scores, multi_class='ovr', average='weighted')
        else: print("Warning: Only one class present. AUC is set to 0.")
    else: precision = f1 = accuracy = auc = 0.0
    recall = recall_score(matched_targets, np.where(np.array(matched_preds) == -1, matched_targets, matched_preds), average='weighted', zero_division=0) if len(matched_targets) > 0 else 0.0
    metrics = {
        'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Accuracy': accuracy, 'AUC': auc,
        'mAP@0.5': float(map_metrics['map_50']), 'mAP@0.5:0.95': float(map_metrics['map']),
        'Test Time (ms/img)': avg_inference_time * 1000
    }
    return metrics, matched_preds, matched_targets, map_metrics

def box_iou_numpy(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2]); rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter; iou = inter / np.clip(union, 1e-6, None)
    return iou

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    mask = np.array(y_pred) != -1
    y_true_filtered, y_pred_filtered = np.array(y_true)[mask], np.array(y_pred)[mask]
    if len(y_true_filtered) == 0: print("Warning: No valid predictions for confusion matrix"); return
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=range(len(class_names)))
    cm_normalized = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Normalized Value'})
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold'); plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_metrics_bar(metrics, save_path):
    plot_metrics = {k: v for k, v in metrics.items() if 'Test Time' not in k}
    time_metric_val = metrics.get('Test Time (ms/img)', 0)
    fig, ax1 = plt.subplots(figsize=(14, 8))
    metric_names, metric_values = list(plot_metrics.keys()), list(plot_metrics.values())
    colors = sns.color_palette("viridis", len(metric_names))
    bars = ax1.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    for bar in bars:
        height = bar.get_height(); ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1.05); ax1.set_ylabel('Score (0-1)', fontsize=14, fontweight='bold', color='darkblue')
    ax1.set_title('Test Set Evaluation Metrics', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--'); plt.xticks(rotation=15, ha='right')
    ax2 = ax1.twinx(); ax2.plot([], []); ax2.set_yticks([])
    ax2.text(1.02, 0.5, f'Test Time:\n{time_metric_val:.2f} ms/img', transform=ax1.transAxes, fontsize=14, fontweight='bold', ha='left', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='orange', alpha=0.3))
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Metrics bar chart saved to {save_path}")

def plot_metrics_radar(metrics, save_path):
    plot_metrics = {k: v for k, v in metrics.items() if k not in ['mAP@0.5:0.95', 'Test Time (ms/img)']}
    categories, values = list(plot_metrics.keys()), list(plot_metrics.values())
    N = len(categories); angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8]); ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=10); ax.set_rlabel_position(0)
    ax.plot(angles, values, 'o-', linewidth=3, color='#4A90E2', label='Test Set', markersize=8); ax.fill(angles, values, alpha=0.25, color='#4A90E2')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7); ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Metrics Radar Chart', size=16, fontweight='bold', pad=30)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Metrics radar chart saved to {save_path}")

def create_metrics_summary(metrics, model_path, data_root, img_size, save_path):
    fig, ax = plt.subplots(figsize=(12, 8)); ax.axis('off')
    title_text = "FASTER R-CNN MODEL EVALUATION SUMMARY"; ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=20, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', ec='black', lw=2))
    info_text = f"Model: {model_path}\nSplit: {Path(save_path).parent.name}\nImage Size: {img_size}"; ax.text(0.5, 0.85, info_text, ha='center', va='top', fontsize=12, family='monospace', bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', ec='black'))
    metrics_title = "METRICS"; ax.text(0.5, 0.72, metrics_title, ha='center', va='top', fontsize=16, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', ec='black', lw=2))
    metrics_text = ""
    for name, value in metrics.items():
        metrics_text += f"{name:20s} {value:.2f} ms\n" if 'Time' in name else f"{name:20s} {value:.4f}  ({value*100:.2f}%)\n"
    ax.text(0.5, 0.55, metrics_text, ha='center', va='center', fontsize=14, family='monospace', bbox=dict(boxstyle='round,pad=1', fc='white', ec='black', lw=1.5))
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, ec='black', lw=3, transform=ax.transAxes); ax.add_patch(rect)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Metrics summary saved to {save_path}")

def save_evaluation_results(metrics, model_path, data_root, img_size, batch_size, conf_thres, save_path):
    with open(save_path, 'w') as f:
        f.write("="*70 + "\nFaster R-CNN Model - Test Set Evaluation Results\n" + "="*70 + "\n")
        f.write(f"Split: {Path(save_path).parent.name}\nModel: {model_path}\nTest Set: {data_root}\n")
        f.write(f"Image Size: {img_size}\nBatch Size: {batch_size}\nConfidence Threshold: {conf_thres}\n\nMetrics:\n")
        for name, value in metrics.items():
            f.write(f"  {name:20s} {value:.2f} ms\n" if 'Time' in name else f"  {name:20s} {value:.4f} ({value*100:.2f}%)\n")
        f.write("="*70 + "\n")
    print(f"Evaluation results saved to {save_path}")

def plot_comparison_chart(all_results, save_path):
    if len(all_results) < 2: return

    metrics_to_compare = [k for k in all_results[0]['metrics'].keys() if 'Test Time' not in k]
    dataset_names = [res['name'] for res in all_results]
    n_datasets = len(dataset_names)
    n_metrics = len(metrics_to_compare)

    fig, ax = plt.subplots(figsize=(18, 10))

    bar_width = 0.8 / n_datasets
    index = np.arange(n_metrics)

    for i, result in enumerate(all_results):
        metric_values = [result['metrics'][m] for m in metrics_to_compare]
        pos = index - (bar_width * (n_datasets-1) / 2) + (i * bar_width)
        bars = ax.bar(pos, metric_values, bar_width, label=dataset_names[i])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, rotation=90, color='white', fontweight='bold')

    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title('Performance Comparison: Standard vs. Robustness Test Set', fontsize=16, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(metrics_to_compare, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nComparison chart saved to {save_path}")

def run_evaluation_on_split(model, device, data_root, split_name, class_names, num_classes, args, output_dir):
    """Loads data for a split, evaluates the model, and saves all results."""
    print("\n" + "="*40)
    print(f"STARTING EVALUATION ON: {split_name.upper()}")
    print("="*40)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = TestDataset(data_root, split_name, class_names, args.img_size)
    if len(dataset) == 0:
        print(f"Warning: No images found for split '{split_name}'. Skipping evaluation.")
        return None

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    (pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, avg_time) = evaluate_model(
        model, loader, device, args.conf_thres
    )
    
    print("\nCalculating metrics...")
    metrics, matched_preds, matched_targets, _ = calculate_metrics(
        pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, num_classes, class_names, avg_time
    )
    
    print("\n" + "-"*30 + f" RESULTS FOR {split_name.upper()} " + "-"*30)
    for name, value in metrics.items():
        if 'Time' in name: print(f"{name:20s} {value:.2f} ms")
        else: print(f"{name:20s} {value:.4f} ({value*100:.2f}%)")
    print("-"*(70))
    
    print("\nGenerating visualizations...")
    plot_confusion_matrix(matched_targets, matched_preds, class_names, save_path=output_dir / 'confusion_matrix.png')
    plot_metrics_bar(metrics, save_path=output_dir / 'metrics_bar_chart.png')
    plot_metrics_radar(metrics, save_path=output_dir / 'metrics_radar_chart.png')
    create_metrics_summary(metrics, args.model, data_root / split_name, args.img_size, save_path=output_dir / 'metrics_summary.png')
    save_evaluation_results(metrics, args.model, data_root / split_name, args.img_size, args.batch_size, args.conf_thres, save_path=output_dir / 'evaluation_results.txt')
    
    return metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    main_output_dir = Path(args.output_dir)
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70 + "\nFASTER R-CNN MODEL EVALUATION\n" + "="*70)
    
    data_root = Path(args.data)
    class_names = load_class_names(data_root)
    num_classes = len(class_names)
    print(f"\nNumber of classes: {num_classes}\nClasses: {class_names}")
    
    print(f"\nLoading model from {args.model}...")
    model = FasterRCNNDetector(num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print("Model loaded successfully")
    
    all_results = []

    standard_output_dir = main_output_dir / 'standard_test'
    standard_metrics = run_evaluation_on_split(
        model, device, data_root, 'test', class_names, num_classes, args, standard_output_dir
    )
    if standard_metrics:
        all_results.append({'name': 'Standard Test', 'metrics': standard_metrics})

    if args.robustness_data:
        robustness_path = Path(args.robustness_data)
        robustness_root = robustness_path.parent
        robustness_split = robustness_path.name
        
        robustness_output_dir = main_output_dir / 'robustness_test'
        robustness_metrics = run_evaluation_on_split(
            model, device, robustness_root, robustness_split, class_names, num_classes, args, robustness_output_dir
        )
        if robustness_metrics:
            all_results.append({'name': f'Robustness Test ({robustness_split})', 'metrics': robustness_metrics})

    if len(all_results) > 1:
        print("\n" + "="*70 + "\nGENERATING FINAL COMPARISON\n" + "="*70)
        plot_comparison_chart(all_results, save_path=main_output_dir / 'comparison_metrics.png')
    
    print(f"\nAll evaluations complete! Results saved to {main_output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster R-CNN Model Evaluation with Robustness Testing')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to the main dataset root directory')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for inference')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--output-dir', type=str, default='test_results_frcnn', help='Directory to save all evaluation results')
    parser.add_argument('--robustness-data', type=str, default=None, help='(Optional) Path to the robustness test set directory (e.g., path/to/severe)')
    
    args = parser.parse_args()
    main(args)

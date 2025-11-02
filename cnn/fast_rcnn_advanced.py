import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image, ImageEnhance
import numpy as np
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from collections import Counter
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_names(data_root: Path) -> List[str]:
    yaml_path = data_root / 'data.yaml'
    if not yaml_path.exists(): raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    with open(yaml_path, 'r') as f: cfg = yaml.safe_load(f)
    if 'names' not in cfg: raise KeyError("'names' key not found in data.yaml")
    names = cfg['names']
    if isinstance(names, dict): names = [names[i] for i in sorted(names.keys())]
    print(f"Loaded {len(names)} classes"); return names

def yolo_to_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    xc, yc, bw, bh = box; x1, y1 = (xc - bw / 2.0) * w, (yc - bh / 2.0) * h
    x2, y2 = (xc + bw / 2.0) * w, (yc + bh / 2.0) * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

class EnhancedDetectionDataset(Dataset):
    def __init__(self, root: Path, split: str, class_names: List[str],
                 imgsz: int = 640, augment: bool = True, imbalance_config: Dict = None):
        self.root = root; self.split = split
        self.img_dir = root / split / "images"; self.lbl_dir = root / split / "labels"
        self.img_paths = sorted([p for p in self.img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.imgsz = imgsz; self.augment = augment and (split == 'train')
        self.class_names = class_names; self.num_classes = len(class_names)
        
        if imbalance_config and split == 'train':
            self.apply_imbalance(imbalance_config)
        
        print(f"Loaded {len(self.img_paths)} images for '{split}' from {self.img_dir}")

    def apply_imbalance(self, config):
        print(f"\n{'='*60}\nCREATING IMBALANCED DATASET (Factor: {config['factor']})\n{'='*60}")
        imbalance_factor = config['factor']; sampled_paths = []
        for img_path in self.img_paths:
            lbl_path = (self.lbl_dir / img_path.name).with_suffix('.txt')
            if lbl_path.exists():
                lines = lbl_path.read_text().strip().splitlines()
                if lines and lines[0].strip():
                    main_label = int(float(lines[0].split()[0]))
                    ratio = np.exp(-main_label * np.log(imbalance_factor) / (self.num_classes - 1 if self.num_classes > 1 else 1))
                    if np.random.rand() < ratio: sampled_paths.append(img_path)
        print(f"Original train size: {len(self.img_paths)}, Imbalanced size: {len(sampled_paths)}")
        self.img_paths = sampled_paths

    def __len__(self): return len(self.img_paths)

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

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]; image = Image.open(img_path).convert('RGB'); W, H = image.size
        lbl_path = (self.lbl_dir / img_path.name).with_suffix('.txt'); boxes, labels = self._read_labels(lbl_path, W, H)
        if self.augment:
            if random.random() < 0.5: image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.5: image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if boxes.numel() > 0: x1 = boxes[:, 0].clone(); boxes[:, 0] = W - boxes[:, 2]; boxes[:, 2] = W - x1
        scale_w, scale_h = self.imgsz / W, self.imgsz / H; image = image.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale_w; boxes[:, [1, 3]] *= scale_h
            boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], 0, self.imgsz); boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], 0, self.imgsz)
        return to_tensor(image), boxes, labels

def collate_fn(batch): images, boxes, labels = zip(*batch); return list(images), list(boxes), list(labels)

class FasterRCNNDetector(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, trainable_backbone_layers: int = 3):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights, trainable_backbone_layers=trainable_backbone_layers)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        print(f"Created Faster R-CNN with ResNet50 backbone (pretrained={pretrained})")
    def forward(self, images, targets=None): return self.model(images, targets)

class DistortionEvaluator:
    def __init__(self, model, device, class_names): self.model, self.device, self.class_names, self.distortion_results = model, device, class_names, {}
    def add_gaussian_noise(self, image: np.ndarray, sigma: float) -> np.ndarray: return np.clip(image + np.random.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)
    def add_gaussian_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray: kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size; return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray: return np.clip(image * factor, 0, 255).astype(np.uint8)
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray: mean = image.mean(); return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    def add_occlusion(self, image: np.ndarray, occlusion_ratio: float) -> np.ndarray:
        h, w = image.shape[:2]; occluded = image.copy(); num_blocks = int(occlusion_ratio * 10)
        for _ in range(num_blocks):
            block_h, block_w = int(h * 0.1), int(w * 0.1)
            if w > block_w and h > block_h: x, y = np.random.randint(0, w - block_w), np.random.randint(0, h - block_h); occluded[y:y+block_h, x:x+block_w] = 128
        return occluded
    def apply_distortion(self, image: np.ndarray, distortion_type: str, level: float) -> np.ndarray:
        if distortion_type == 'gaussian_noise': return self.add_gaussian_noise(image, sigma=level)
        elif distortion_type == 'gaussian_blur': return self.add_gaussian_blur(image, kernel_size=int(level))
        elif distortion_type == 'low_brightness': return self.adjust_brightness(image, factor=level)
        elif distortion_type == 'low_contrast': return self.adjust_contrast(image, factor=level)
        elif distortion_type == 'occlusion': return self.add_occlusion(image, occlusion_ratio=level)
        return image
    def _evaluate_dataloader(self, dataloader, distortion_type, level, metric_func):
        self.model.eval(); metric = MeanAveragePrecision(box_format='xyxy')
        for images, boxes, labels in tqdm(dataloader, desc=f"Evaluating Level {level}"):
            distorted_images = []
            for img_tensor in images:
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                if distortion_type is not None: img_np = self.apply_distortion(img_np, distortion_type, level)
                distorted_images.append(to_tensor(img_np).to(self.device))
            with torch.no_grad(): predictions = self.model(distorted_images)
            targets, preds = [], []
            for b, l in zip(boxes, labels): targets.append({'boxes': b.cpu(), 'labels': l.cpu()})
            for pred in predictions: preds.append({'boxes': pred['boxes'].cpu(), 'labels': (pred['labels'] - 1).cpu(), 'scores': pred['scores'].cpu()})
            metric.update(preds, targets)
        return {k: float(v) for k, v in metric.compute().items() if k.startswith('map')}
    def evaluate_with_distortion(self, dataloader, distortion_type: str, distortion_levels: List[float]):
        results = {}; print(f"\n{'='*60}\nEvaluating {distortion_type} robustness\n{'='*60}")
        print("\n[Baseline] No distortion..."); baseline_metric = self._evaluate_dataloader(dataloader, None, 0, None); results[0] = baseline_metric
        for level in distortion_levels: print(f"\n[{distortion_type}] Level: {level}"); results[level] = self._evaluate_dataloader(dataloader, distortion_type, level, None)
        self.distortion_results[distortion_type] = results
    def plot_robustness_curves(self, save_dir: Path):
        num_distortions = len(self.distortion_results); fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.flatten()
        for idx, (dist_type, results) in enumerate(self.distortion_results.items()):
            ax = axes[idx]; levels = sorted(results.keys()); maps = [results[level].get('map', 0) for level in levels]
            ax.plot(levels, maps, marker='o'); ax.set_xlabel('Distortion Level'); ax.set_ylabel('mAP')
            ax.set_title(f'{dist_type.replace("_", " ").title()} Robustness'); ax.grid(True, alpha=0.3)
        for idx in range(num_distortions, len(axes)): axes[idx].axis('off')
        plt.tight_layout(); plt.savefig(save_dir / 'robustness_curves.png', dpi=150); plt.close()
        print(f"✓ Robustness curves saved to {save_dir}")
    def print_summary(self):
        print(f"\n{'='*70}\nROBUSTNESS EVALUATION SUMMARY\n{'='*70}")
        for dist_type, results in self.distortion_results.items():
            print(f"\n{dist_type.upper().replace('_', ' ')}:"); baseline_map = results[0].get('map', 0)
            print(f"  Baseline (no distortion): mAP = {baseline_map:.4f}")
            for level, metrics in sorted(results.items())[1:]:
                current_map = metrics.get('map', 0)
                degradation = ((baseline_map - current_map) / baseline_map) * 100 if baseline_map > 0 else 0
                print(f"  Level {level}: mAP = {current_map:.4f} (↓ {degradation:.1f}%)")
    def save_results(self, save_dir: Path):
        with open(save_dir / 'robustness_metrics.json', 'w') as f: json.dump(self.distortion_results, f, indent=4)
        print(f"✓ Robustness metrics saved to {save_dir}")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.target_layer = model, target_layer; self.gradients, self.activations = None, None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]
    def __call__(self, x, index=None):
        self.model.eval()
        output = self.model(x)
        if index is None:
            scores = torch.cat([p['scores'] for p in output])
            if len(scores) == 0: return None
            index = scores.argmax()
        
        loss = sum([p['scores'].sum() for p in output])
        self.model.zero_grad(); loss.backward()
        
        gradients = self.gradients.cpu().numpy()[0]; activations = self.activations.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2)); cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights): cam += w * activations[i, :, :]
        cam = np.maximum(cam, 0); cam = cv2.resize(cam, (x.shape[3], x.shape[2])); cam -= np.min(cam); cam /= np.max(cam)
        return cam

def create_failure_analysis(model, dataloader, device, class_names, save_dir, num_failures=20):
    model.eval(); failure_cases = []
    with torch.no_grad():
        for images, boxes, labels in tqdm(dataloader, desc="Finding failure cases"):
            images_list = [img.to(device) for img in images]; predictions = model(images_list)
            for img, pred, gt_box, gt_label in zip(images, predictions, boxes, labels):
                if len(gt_label) > 0 and (len(pred['scores']) == 0 or pred['scores'].max() < 0.5):
                    failure_cases.append({'image': img, 'prediction': pred, 'gt_boxes': gt_box, 'gt_labels': gt_label, 'failure_type': 'missed/low_conf'})
                    if len(failure_cases) >= num_failures: break
            if len(failure_cases) >= num_failures: break
    for idx, case in enumerate(failure_cases):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8)); img_np = (case['image'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8); ax.imshow(img_np)
        for box, label in zip(case['gt_boxes'], case['gt_labels']):
            box = box.cpu().numpy(); rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect); ax.text(box[0], box[1]-5, f"GT: {class_names[label]}", color='g')
        pred = case['prediction']
        if len(pred['scores']) > 0:
            for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                box = box.cpu().numpy(); rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
                ax.add_patch(rect); ax.text(box[0], box[3]+15, f"Pred: {class_names[label-1]} ({score:.2f})", color='r')
        ax.set_title(f"Failure Case {idx+1}: {case['failure_type']}"); ax.axis('off'); plt.tight_layout()
        plt.savefig(save_dir / f'failure_{idx:03d}.png', dpi=150); plt.close()
    print(f"✓ Saved {len(failure_cases)} failure case analyses to {save_dir}")

def train_one_epoch(model, optimizer, dataloader, device, epoch, scaler, writer, class_weights=None):
    model.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
    total_loss = 0.0
    
    for i, (images, boxes, labels) in loop:
        images = [img.to(device) for img in images]
        targets = []
        for b, l in zip(boxes, labels):
            target = {'boxes': b.to(device), 'labels': l.to(device) + 1}
            targets.append(target)
        
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=scaler is not None):
            loss_dict = model(images, targets)
            
            if class_weights is not None and 'loss_classifier' in loss_dict:
                batch_classes = torch.cat([t['labels'] for t in targets]).unique() - 1
                if batch_classes.numel() > 0:
                    weights_for_batch = class_weights[batch_classes.long()].mean()
                    loss_dict['loss_classifier'] *= weights_for_batch
            
            losses = sum(loss for loss in loss_dict.values())
        
        if scaler:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        
        total_loss += losses.item()
        loop.set_postfix(loss=losses.item())
        
        if writer and i % 50 == 0:
            writer.add_scalar('Loss/train_iter', losses.item(), epoch * len(dataloader) + i)
            for k, v in loss_dict.items():
                writer.add_scalar(f'Loss/{k}', v.item(), epoch * len(dataloader) + i)
                
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy')
    
    for images, boxes, labels in tqdm(dataloader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        predictions = model(images)
        targets, preds = [], []
        
        for b, l in zip(boxes, labels):
            targets.append({'boxes': b.cpu(), 'labels': l.cpu()})
            
        for pred in predictions:
            preds.append({
                'boxes': pred['boxes'].cpu(),
                'labels': (pred['labels'] - 1).cpu(),
                'scores': pred['scores'].cpu()
            })
            
        metric.update(preds, targets)
        
    metrics_dict = metric.compute()
    return {k: float(v) for k, v in metrics_dict.items() if k.startswith('map')}

def main():
    parser = argparse.ArgumentParser(description='Enhanced Faster R-CNN Training (37+ Features)')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='runs/faster_rcnn_advanced')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--create_imbalance', action='store_true')
    parser.add_argument('--imbalance_factor', type=float, default=10.0)
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--eval_robustness', action='store_true')
    parser.add_argument('--eval_explainability', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}"); data_root = Path(args.data_root); save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'robustness').mkdir(exist_ok=True); (save_dir / 'explainability').mkdir(exist_ok=True, parents=True)
    class_names = load_names(data_root); num_classes = len(class_names)
    
    imbalance_config = {'factor': args.imbalance_factor} if args.create_imbalance else None
    train_dataset = EnhancedDetectionDataset(data_root, 'train', class_names, args.imgsz, augment=True, imbalance_config=imbalance_config)
    val_dataset = EnhancedDetectionDataset(data_root, 'valid', class_names, args.imgsz, augment=False)
    test_dataset = EnhancedDetectionDataset(data_root, 'test', class_names, args.imgsz, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    class_weights = None
    if args.use_class_weights:
        print("Computing class weights..."); all_labels = []
        for i in tqdm(range(len(train_dataset)), desc="Collecting labels"):
            _, _, labels = train_dataset[i]
            if len(labels) > 0: all_labels.extend(labels.tolist())
        counts = Counter(all_labels); total = sum(counts.values())
        class_weights = torch.tensor([total / (counts.get(i, 1e-6) * num_classes) for i in range(num_classes)]).to(device)
        print(f"Class weights: {class_weights.cpu().numpy().round(2)}")

    model = FasterRCNNDetector(num_classes, pretrained=args.pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler('cuda') if args.amp else None
    writer = SummaryWriter(str(save_dir / 'logs')) if args.tensorboard else None
    
    best_map = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}\n{'='*60}")
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler, writer, class_weights)
        metrics = evaluate(model, val_loader, device)
        current_map = metrics.get('map', 0.0)
        print(f"Training Loss: {train_loss:.4f}, Validation mAP: {current_map:.4f}")
        if writer: writer.add_scalar('Loss/train_epoch', train_loss, epoch); writer.add_scalar('mAP/val', current_map, epoch)
        scheduler.step(current_map)
        if current_map > best_map:
            best_map = current_map; torch.save(model.state_dict(), save_dir / 'best.pt'); print(f"✓ Saved new best model (mAP: {best_map:.4f})")
    
    if writer: writer.close()
    print(f"\n{'='*60}\nTraining completed! Best mAP: {best_map:.4f}\n{'='*60}\n")
    
    print("Loading best model for advanced evaluation...")
    model.load_state_dict(torch.load(save_dir / 'best.pt'))

    if args.eval_robustness:
        print(f"\n--- Starting Robustness Evaluation ---")
        evaluator = DistortionEvaluator(model, device, class_names)
        distortion_configs = {'gaussian_noise': [10, 20, 30], 'gaussian_blur': [3, 5, 7], 'low_brightness': [0.7, 0.5, 0.3]}
        for dist_type, levels in distortion_configs.items():
            evaluator.evaluate_with_distortion(test_loader, dist_type, levels)
        evaluator.print_summary()
        evaluator.plot_robustness_curves(save_dir=(save_dir / 'robustness'))
        evaluator.save_results(save_dir=(save_dir / 'robustness'))

    if args.eval_explainability:
        print(f"\n--- Starting Explainability Analysis ---")
        create_failure_analysis(model, test_loader, device, class_names, save_dir=(save_dir / 'explainability' / 'failures'))
        
        try:
            target_layer = model.model.backbone.body.layer4[-1]
            grad_cam = GradCAM(model, target_layer)
            print("Grad-CAM initialized. Generating samples...")
            count = 0
            for img_tensor, _, _ in test_loader:
                if count >= 5: break
                cam_map = grad_cam(img_tensor[0].unsqueeze(0).to(device))
                if cam_map is not None:
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1); plt.imshow(img_tensor[0].permute(1,2,0)); plt.title('Original'); plt.axis('off')
                    plt.subplot(1, 2, 2); plt.imshow(img_tensor[0].permute(1,2,0)); plt.imshow(cam_map, cmap='jet', alpha=0.5); plt.title('Grad-CAM'); plt.axis('off')
                    plt.savefig(save_dir / 'explainability' / f'gradcam_{count}.png'); plt.close()
                    count += 1
            print(f"Generated {count} Grad-CAM samples.")
        except Exception as e: print(f"Could not generate Grad-CAM: {e}")

if __name__ == '__main__':
    main()

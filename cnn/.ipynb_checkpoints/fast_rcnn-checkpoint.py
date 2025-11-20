import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

import torchvision
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Optional metrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms.functional import to_tensor

from tqdm import tqdm

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_names(data_root: Path) -> List[str]:
    yaml_path = data_root / 'data.yaml'
    print(yaml_path)
    if yaml_path.exists():
        cfg = yaml.safe_load(open(yaml_path))
        names = cfg['names']
        return names


def yolo_to_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert YOLO (xc,yc,w,h in [0,1]) -> absolute [x1,y1,x2,y2]."""
    xc, yc, bw, bh = box
    xc *= w
    yc *= h
    bw *= w
    bh *= h
    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def hflip(image: Image.Image, boxes: torch.Tensor) -> Tuple[Image.Image, torch.Tensor]:
    w, _ = image.size
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if boxes.numel() > 0:
        x1 = boxes[:, 0].clone()
        x2 = boxes[:, 2].clone()
        boxes[:, 0] = w - x2
        boxes[:, 2] = w - x1
    return image, boxes


def resize_short_side(image: Image.Image, boxes: torch.Tensor, target_short: int) -> Tuple[Image.Image, torch.Tensor]:
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


class YoloDetectionDataset(Dataset):
    def __init__(self, root: Path, split: str, class_names: List[str], imgsz: int = 1024, hflip_p: float = 0.5):
        self.root = Path(root)
        self.split = split
        self.class_names = class_names

        self.img_dir = self.root / self.split / "images"
        self.lbl_dir = self.root / self.split / "labels"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {self.img_dir}")
        if not self.lbl_dir.exists():
            raise FileNotFoundError(f"Label folder not found: {self.lbl_dir}")

        self.img_paths = sorted([
            p for p in self.img_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        self.imgsz = imgsz
        self.hflip_p = hflip_p


    def __len__(self) -> int:
        return len(self.img_paths)

    def _read_labels(self, lbl_path: Path, W: int, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        W, H = image.size

        if random.random() < self.hflip_p:
            image, boxes = hflip(image, boxes)

        image_t = to_tensor(image)

        target: Dict[str, torch.Tensor] = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'iscrowd': torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes.numel() > 0 else torch.zeros((0,), dtype=torch.int64),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.numel() > 0 else torch.zeros((0,), dtype=torch.float32)
        }
        
        return image_t, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT' if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes + 1)
    return model


def train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=10):
    model.train()
    running = 0.0
    for i, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running += losses.item()
        if (i + 1) % print_freq == 0:
            avg = running / print_freq
            print(f"Epoch {epoch} | Iter {i+1}/{len(dataloader)} | loss {avg:.4f} | components: " + 
                  ', '.join([f"{k}:{v.item():.3f}" for k, v in loss_dict.items()]))
            running = 0.0


def evaluate_loss(model, dataloader, device) -> float:
    model.train()
    total, n = 0.0, 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            total += loss.item()
            n += 1
    return total / max(1, n)


def evaluate_map(model, dataloader, device, iou_type: str = 'bbox') -> dict:
    """Compute a few simple scalar metrics (mAP/mAR) using torchmetrics."""
    """Returns only scalar keys to keep it simple."""
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', iou_type=iou_type, class_metrics=False)
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
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
    res = metric.compute()
    
    keys = ['map', 'map_50', 'map_75', 'mar_100']
    out = {}
    for k in keys:
        if k in res:
            v = res[k]
            if isinstance(v, torch.Tensor):
                out[k] = float(v.item())
            else:
                out[k] = float(v)
    return out


def _greedy_match_iou(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """Return matches: indices of pred matched to gt using greedy IoU matching."""
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return set(), set(range(len(pred_boxes))), set(range(len(gt_boxes)))
    ious = box_iou(pred_boxes, gt_boxes).numpy()
    matched_pred, matched_gt = set(), set()
    
    order = np.dstack(np.unravel_index(np.argsort(-ious.ravel()), ious.shape))[0]
    for i, j in order:
        if ious[i, j] < iou_thresh:
            break
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(int(i))
        matched_gt.add(int(j))
    fp = set(range(len(pred_boxes))) - matched_pred
    fn = set(range(len(gt_boxes))) - matched_gt
    return matched_pred, fp, fn


def evaluate_prf1(model, dataloader, device, iou_thresh=0.5, score_thr=0.5, num_classes: int = None) -> dict:
    """Compute per-class Precision/Recall/F1 by thresholding scores and greedy IoU matching."""
    """Background is excluded; classes are assumed 1..num_classes."""
    model.eval()
    if num_classes is None:
        raise ValueError('num_classes required for PR/F1 evaluation')
    TP = np.zeros(num_classes + 1, dtype=np.int64)
    FP = np.zeros(num_classes + 1, dtype=np.int64)
    FN = np.zeros(num_classes + 1, dtype=np.int64)
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                
                mask = out['scores'].detach().cpu() >= score_thr
                pboxes = out['boxes'].detach().cpu()[mask]
                plabels = out['labels'].detach().cpu()[mask]
                
                gboxes = tgt['boxes'].detach().cpu()
                glabels = tgt['labels'].detach().cpu()
                
                for c in range(1, num_classes + 1):
                    p_idx = (plabels == c).nonzero(as_tuple=False).flatten()
                    g_idx = (glabels == c).nonzero(as_tuple=False).flatten()
                    pb = pboxes[p_idx]
                    gb = gboxes[g_idx]
                    matched_pred, fp_set, fn_set = _greedy_match_iou(pb, None, gb, iou_thresh=iou_thresh)
                    TP[c] += len(matched_pred)
                    FP[c] += len(fp_set)
                    FN[c] += len(fn_set)
    
    per_class = {}
    for c in range(1, num_classes + 1):
        tp, fp, fn = TP[c], FP[c], FN[c]
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = 2 * prec * rec / max(1e-8, (prec + rec)) if (prec + rec) > 0 else 0.0
    
    tp_sum, fp_sum, fn_sum = TP.sum(), FP.sum(), FN.sum()
    micro_p = tp_sum / max(1, (tp_sum + fp_sum))
    micro_r = tp_sum / max(1, (tp_sum + fn_sum))
    micro_f1 = 2 * micro_p * micro_r / max(1e-8, (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    return {
        'per_class': per_class,
        'micro': {'precision': float(micro_p), 'recall': float(micro_r), 'f1': float(micro_f1)},
    }


def evaluate_pr_auc(model, dataloader, device, iou_thresh=0.5, num_classes: int = None, thresholds: int = 101) -> dict:
    """Compute PR-AUC by sweeping score thresholds; returns micro PR-AUC."""
    """Note: AUC-ROC is not well-defined for detection; PR-AUC is more meaningful."""
    model.eval()
    if num_classes is None:
        raise ValueError('num_classes required for PR-AUC evaluation')
    
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                all_preds.append({
                    'boxes': out['boxes'].detach().cpu(),
                    'scores': out['scores'].detach().cpu(),
                    'labels': out['labels'].detach().cpu(),
                })
                all_gts.append({
                    'boxes': tgt['boxes'].detach().cpu(),
                    'labels': tgt['labels'].detach().cpu(),
                })
    
    thrs = np.linspace(0.0, 1.0, thresholds)
    precs, recs = [], []
    for thr in thrs:
        TP=FP=FN=0
        for pred, gt in zip(all_preds, all_gts):
            mask = pred['scores'] >= thr
            pb = pred['boxes'][mask]
            pl = pred['labels'][mask]
            gb = gt['boxes']
            gl = gt['labels']
            
            for c in range(1, num_classes + 1):
                p_idx = (pl == c).nonzero(as_tuple=False).flatten()
                g_idx = (gl == c).nonzero(as_tuple=False).flatten()
                mp, fp, fn = _greedy_match_iou(pb[p_idx], None, gb[g_idx], iou_thresh=iou_thresh)
                TP += len(mp); FP += len(fp); FN += len(fn)
        p = TP / max(1, (TP + FP))
        r = TP / max(1, (TP + FN))
        precs.append(p); recs.append(r)
    
    order = np.argsort(recs)
    recs = np.array(recs)[order]
    precs = np.array(precs)[order]
    
    pr_auc = float(np.trapezoid(precs, recs))
    return {'pr_auc_micro': pr_auc, 'pr_curve': {'recall': recs.tolist(), 'precision': precs.tolist()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset', help='Dataset root directory')
    parser.add_argument('--epochs', type=int, default=20, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW optimizer')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes for dataloader')
    parser.add_argument('--imgsz', type=int, default=1024, help='Short side dimension for image scaling')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Training device (cuda or cpu)')
    parser.add_argument('--no_pretrain', action='store_true', help='Do not use a pretrained model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='(not used in this script) validation set ratio')
    parser.add_argument('--save_dir', type=str, default='./runs_frcnn', help='Directory to save models')
    parser.add_argument('--eval_map', action='store_true', help='Compute mAP/mAR after each epoch')
    parser.add_argument('--eval_prf1', action='store_true', help='Compute P/R/F1 after each epoch')
    parser.add_argument('--eval_prauc', action='store_true', help='Compute PR-AUC (micro) after training')
    parser.add_argument('--iou_eval', type=float, default=0.5, help='IoU threshold for PR/F1 and PR-AUC')
    parser.add_argument('--conf_thr', type=float, default=0.05, help='Confidence threshold for PR/F1')
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root)

    class_names = load_names(data_root)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    train_ds = YoloDetectionDataset(data_root, 'train', class_names, imgsz=args.imgsz)
    val_ds = YoloDetectionDataset(data_root, 'valid', class_names, imgsz=args.imgsz, hflip_p=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    device = torch.device(args.device)
    model = create_model(num_classes=num_classes, pretrained=not args.no_pretrain)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float('inf')
    no_improve = 0
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate_loss(model, val_loader, device)
        lr_scheduler.step()
        print(f"Epoch {epoch} | val_loss {val_loss:.4f}")
        
        if args.eval_map:
            m = evaluate_map(model, val_loader, device)
            if m:
                print('[mAP/mAR]', {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if isinstance(v, (float, int))})
        if args.eval_prf1:
            prf1 = evaluate_prf1(model, val_loader, device, iou_thresh=args.iou_eval, score_thr=args.conf_thr, num_classes=num_classes)
            print('[P/R/F1 micro]', {k: round(v, 4) for k, v in prf1['micro'].items()})
        
        ckpt_path = Path(args.save_dir) / f"epoch{epoch:03d}_val{val_loss:.4f}.pth"
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss, 'classes': class_names}, ckpt_path)
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss, 'classes': class_names}, Path(args.save_dir) / 'best.pth')
            print(f"Saved best checkpoint: val_loss={val_loss:.4f}")
        else:
            print("No improvement.")
            no_improve += 1
        
        if no_improve >= 10:
            print("Early stopping due to no improvement.")
            break

    if args.eval_prauc:
        pra = evaluate_pr_auc(model, val_loader, device, iou_thresh=args.iou_eval, num_classes=num_classes)
        print('[PR-AUC micro]', pra.get('pr_auc_micro', None))

    print('Training complete.')


if __name__ == '__main__':
    main()
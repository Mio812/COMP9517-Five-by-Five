import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict
import math

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image, ImageEnhance
import numpy as np

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# ============================================================
# ⚙️ UTILS & SETUP
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_names(data_root: Path) -> List[str]:
    yaml_path = data_root / 'data.yaml'
    if not yaml_path.exists(): 
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    with open(yaml_path, 'r') as f: 
        cfg = yaml.safe_load(f)
    if 'names' not in cfg: 
        raise KeyError("'names' key not found in data.yaml")
    names = cfg['names']
    if isinstance(names, dict): 
        names = [names[i] for i in sorted(names.keys())]
    print(f"Loaded {len(names)} classes")
    return names

def yolo_to_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    """Convert YOLO format (xc, yc, w, h) to xyxy format"""
    xc, yc, bw, bh = box
    x1, y1 = (xc - bw / 2.0) * w, (yc - bh / 2.0) * h
    x2, y2 = (xc + bw / 2.0) * w, (yc + bh / 2.0) * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

# ============================================================
# 📦 DATASET
# ============================================================

class DetectionDataset(Dataset):
    """Dataset for object detection with YOLO format labels"""
    def __init__(self, root: Path, split: str, class_names: List[str], 
                 imgsz: int = 640, hflip_p: float = 0.5, strong_aug: bool = True):
        self.img_dir = root / split / "images"
        self.lbl_dir = root / split / "labels"
        self.img_paths = sorted([p for p in self.img_dir.rglob("*") 
                                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.imgsz = imgsz
        self.hflip_p = hflip_p
        self.strong_aug = strong_aug and (split == 'train')
        self.class_names = class_names
        self.num_classes = len(class_names)
        print(f"Loaded {len(self.img_paths)} images from {self.img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def _read_labels(self, lbl_path: Path, W: int, H: int):
        """Read labels from YOLO format txt file"""
        boxes, labels = [], []
        if not lbl_path.exists():
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)
        
        lines = [ln.strip() for ln in lbl_path.read_text().strip().splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5: 
                continue
            cid = int(float(parts[0]))
            if not (0 <= cid < self.num_classes):
                print(f"Warning: Invalid class ID {cid} in {lbl_path}. Skipping. Max valid ID is {self.num_classes-1}.")
                continue
            xyxy = yolo_to_xyxy(np.array(list(map(float, parts[1:5]))), W, H)
            # Filter out invalid boxes
            if (xyxy[2] - xyxy[0]) > 1 and (xyxy[3] - xyxy[1]) > 1:
                boxes.append(xyxy.tolist())
                labels.append(cid)
        
        if not boxes:
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        W, H = image.size
        
        lbl_path = (self.lbl_dir / img_path.name).with_suffix('.txt')
        boxes, labels = self._read_labels(lbl_path, W, H)

        # Strong augmentation (only for training)
        if self.strong_aug:
            if random.random() < 0.5: 
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.5: 
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))

        # Horizontal flip
        if random.random() < self.hflip_p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                boxes[:, 0] = W - boxes[:, 2]
                boxes[:, 2] = W - x1

        # Resize image and boxes
        scale_w, scale_h = self.imgsz / W, self.imgsz / H
        image = image.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale_w
            boxes[:, [1, 3]] *= scale_h
            # Clamp boxes to image boundaries
            boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], 0, self.imgsz)
            boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], 0, self.imgsz)

        return to_tensor(image), boxes, labels

def collate_fn(batch):
    """Custom collate function for variable number of boxes per image"""
    images, boxes, labels = zip(*batch)
    return list(images), list(boxes), list(labels)

# ============================================================
# 🧠 MODEL ARCHITECTURE (FASTER R-CNN)
# ============================================================

class FasterRCNNDetector(nn.Module):
    """Faster R-CNN detector with customizable backbone"""
    def __init__(self, num_classes: int, backbone_name: str = 'resnet50', 
                 pretrained: bool = True, trainable_backbone_layers: int = 3):
        super().__init__()
        self.num_classes = num_classes
        
        if backbone_name == 'resnet50':
            # Use pretrained Faster R-CNN with ResNet50 FPN backbone
            self.model = fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
                trainable_backbone_layers=trainable_backbone_layers,
                box_score_thresh=0.05,  # Lower threshold for more detections
                box_nms_thresh=0.5
            )
            
            # Replace the classifier head
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
            
        elif backbone_name == 'resnet101':
            # Use ResNet101 backbone
            from torchvision.models import resnet101
            from torchvision.models.detection.backbone_utils import BackboneWithFPN
            
            backbone = resnet101(pretrained=pretrained)
            # Remove the final FC layer
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            
            # Create FPN
            backbone = BackboneWithFPN(
                backbone,
                return_layers={'4': '0', '5': '1', '6': '2', '7': '3'},
                in_channels_list=[256, 512, 1024, 2048],
                out_channels=256
            )
            
            # Create Faster R-CNN with custom backbone
            anchor_generator = AnchorGenerator(
                sizes=((32,), (64,), (128,), (256,), (512,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5
            )
            
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2
            )
            
            self.model = FasterRCNN(
                backbone,
                num_classes=num_classes + 1,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                box_score_thresh=0.05,
                box_nms_thresh=0.5
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        print(f"Created Faster R-CNN with {backbone_name} backbone")

    def forward(self, images, targets=None):
        """
        Forward pass
        Args:
            images: list of tensors or tensor of shape [B, 3, H, W]
            targets: list of dicts with 'boxes' and 'labels' keys (for training)
        """
        if self.training:
            # Training mode: return losses
            return self.model(images, targets)
        else:
            # Evaluation mode: return predictions
            return self.model(images)

# ============================================================
# 🔄 TRAINING & EVALUATION
# ============================================================

def train_one_epoch(model, optimizer, dataloader, device, epoch, scaler, writer):
    """Train for one epoch"""
    model.train()
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
    total_loss = 0.0
    
    for i, (images, boxes, labels) in loop:
        # Move to device
        images = [img.to(device) for img in images]
        
        # Prepare targets for Faster R-CNN format
        targets = []
        for b, l in zip(boxes, labels):
            target = {}
            target['boxes'] = b.to(device) if b.numel() > 0 else torch.zeros((0, 4), device=device)
            target['labels'] = l.to(device) + 1 if l.numel() > 0 else torch.zeros((0,), dtype=torch.int64, device=device)  # Add 1 because 0 is background
            targets.append(target)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast('cuda', enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        if scaler:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += losses.item()
        loop.set_postfix(loss=losses.item())
        
        # TensorBoard logging
        if writer and i % 50 == 0:
            writer.add_scalar('Loss/train_iter', losses.item(), epoch * len(dataloader) + i)
            for k, v in loss_dict.items():
                writer.add_scalar(f'Loss/{k}', v.item(), epoch * len(dataloader) + i)
    
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device, class_names):
    """Evaluate model on validation set"""
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy')
    
    for images, boxes, labels in tqdm(dataloader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        
        # Get predictions
        predictions = model(images)
        
        # Prepare targets and predictions for metric calculation
        targets = []
        preds = []
        
        for b, l in zip(boxes, labels):
            target = {
                'boxes': b.cpu(),
                'labels': l.cpu()
            }
            targets.append(target)
        
        for pred in predictions:
            # Convert labels back (subtract 1 as we added 1 during training)
            pred_dict = {
                'boxes': pred['boxes'].cpu(),
                'labels': (pred['labels'] - 1).cpu(),
                'scores': pred['scores'].cpu()
            }
            preds.append(pred_dict)
        
        metric.update(preds, targets)
    
    # Compute metrics
    metrics_dict = metric.compute()
    return {k: float(v) for k, v in metrics_dict.items() if k.startswith('map')}

# ============================================================
# 🚀 MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN Detector Training')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--save_dir', type=str, default='runs/faster_rcnn', help='Save directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet101'], help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
    parser.add_argument('--trainable_backbone_layers', type=int, default=3, 
                        help='Number of trainable backbone layers (3-5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--tensorboard', action='store_true', help='Enable tensorboard logging')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class names
    class_names = load_names(data_root)
    num_classes = len(class_names)
    
    # Create datasets
    train_dataset = DetectionDataset(
        data_root, 'train', class_names, args.imgsz, 
        hflip_p=0.5, strong_aug=True
    )
    val_dataset = DetectionDataset(
        data_root, 'valid', class_names, args.imgsz, 
        hflip_p=0.0, strong_aug=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    
    # Create model
    print(f"Creating Faster R-CNN model (backbone: {args.backbone}, pretrained: {args.pretrained})")
    model = FasterRCNNDetector(
        num_classes=num_classes,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        trainable_backbone_layers=args.trainable_backbone_layers
    ).to(device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if args.amp else None
    
    # TensorBoard writer
    writer = SummaryWriter(save_dir / 'logs') if args.tensorboard else None
    
    # Resume from checkpoint
    start_epoch = 1
    best_map = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint.get('best_map', 0.0)
        print(f"Resumed from epoch {start_epoch-1}, best mAP: {best_map:.4f}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}\n{'='*60}")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler, writer)
        print(f"\nTraining Loss: {train_loss:.4f}")
        
        # Evaluate
        print("\nEvaluating on validation set...")
        metrics = evaluate(model, val_loader, device, class_names)
        current_map = metrics.get('map', 0.0)
        print(f"\nValidation mAP: {current_map:.4f}")
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('mAP/val', current_map, epoch)
            for k, v in metrics.items():
                writer.add_scalar(f'Metrics/{k}', v, epoch)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(current_map)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_map': max(best_map, current_map),
            'metrics': metrics
        }
        torch.save(checkpoint, save_dir / 'last.pt')
        
        # Save best model
        if current_map > best_map:
            best_map = current_map
            torch.save(checkpoint, save_dir / 'best.pt')
            print(f"✓ Saved new best model (mAP: {best_map:.4f})")
    
    if writer:
        writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best mAP: {best_map:.4f}")
    print(f"Models saved to: {save_dir}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

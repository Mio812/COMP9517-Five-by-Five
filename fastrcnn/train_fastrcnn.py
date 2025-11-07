#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fastrcnn.py
=========================
Train a Faster R-CNN model on a YOLO-format insect detection dataset
✅ 支持 Early Stopping
✅ 自动过滤空标签图片
✅ 正确的验证阶段 loss 计算（不会报 list.values() 错误）
"""

import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import functional as F
from models.faster_rcnn_model import create_faster_rcnn


# ============================================================
# 🧠 Dataset Loader for YOLO-format data
# ============================================================
class YoloDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, class_names):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.images = sorted(self.image_dir.glob("*.jpg"))
        self.class_names = class_names

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Skipping {img_path.name}: image file unreadable.")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes, labels = [], []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"⚠️ Invalid label format in {label_path.name}, skipping line.")
                        continue
                    cls, xc, yc, bw, bh = map(float, parts)
                    x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
                    x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)  # background = 0
        else:
            print(f"⚠️ Skipping {img_path.name}: no label file found.")
            return None

        if len(boxes) == 0:
            print(f"⚠️ Skipping {img_path.name}: empty label file (no boxes).")
            return None

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }
        img = F.to_tensor(img)
        return img, target

    def __len__(self):
        return len(self.images)


# ============================================================
# 🧩 Training Function
# ============================================================
def train_fastrcnn(data_dir="dataset", epochs=10, batch_size=4, lr=0.005,
                   save_path="results/fastrcnn_best.pth", class_names=None, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    data_dir = Path(data_dir)
    train_img_dir = data_dir / "train/images"
    train_label_dir = data_dir / "train/labels"
    val_img_dir = data_dir / "valid/images"
    val_label_dir = data_dir / "valid/labels"

    train_dataset = YoloDetectionDataset(train_img_dir, train_label_dir, class_names)
    val_dataset = YoloDetectionDataset(val_img_dir, val_label_dir, class_names)

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return [], []
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    num_classes = len(class_names) + 1  # + background
    model = create_faster_rcnn(num_classes=num_classes, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_loss = float("inf")
    train_losses, val_losses = [], []
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # --- Training ---
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            if len(imgs) == 0:
                continue
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        if len(train_loader) == 0:
            print("❌ No valid training samples found. Please check your dataset labels.")
            return

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.train()  # ✅ 保持train模式以便计算loss
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                if len(imgs) == 0:
                    continue
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        print(f"📉 Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Early Stopping ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve_count = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model (Val Loss={avg_val_loss:.4f}) → {save_path}")
        else:
            no_improve_count += 1
            print(f"⚠️ No improvement for {no_improve_count} epoch(s).")

        if no_improve_count >= patience:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs).")
            break

        lr_scheduler.step()

    # --- Plot Loss Curve ---
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Faster R-CNN Training Curve (Filtered + Early Stopping)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_path).parent / "training_curve.png")
    plt.show()

    print("🎉 Training completed!")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Model saved at: {save_path}")


# ============================================================
# 🖥️ Command Line Interface
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on YOLO-format insect dataset with Early Stopping.")
    parser.add_argument("--data_dir", type=str, default="../dataset", help="Root directory of YOLO-format dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="results/fastrcnn_best.pth", help="Path to save model weights")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (epochs without improvement)")
    parser.add_argument(
        "--class_names",
        nargs="+",
        default=[
            "Ants", "Bees", "Beetles", "Caterpillars",
            "Earthworms", "Earwigs", "Grasshoppers", "Moths",
            "Slugs", "Snails", "Wasps", "Weevils",
        ],
        help="List of class names (excluding background)"
    )

    args = parser.parse_args()
    train_fastrcnn(**vars(args))

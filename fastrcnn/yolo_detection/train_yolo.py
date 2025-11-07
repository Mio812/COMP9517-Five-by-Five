#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_yolo.py
=========================
Train a YOLOv8 model for insect detection.

Usage Example:
--------------
python yolo_detection/train_yolo.py \
    --data_cfg configs/insects.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 8 \
    --device mps \
    --name insect_yolo
"""

from ultralytics import YOLO
import argparse
import os


# ============================================================
# 🧩 Training Function
# ============================================================
def train_yolo(data_cfg, model, epochs, imgsz, batch, device, name):
    """
    Train a YOLOv8 model on the insect dataset.

    Args:
        data_cfg (str): Path to YOLO data YAML file.
        model (str): Model architecture or pretrained weights (e.g. yolov8n.pt).
        epochs (int): Training epochs.
        imgsz (int): Image input size.
        batch (int): Batch size.
        device (str): Training device (cpu, cuda, mps).
        name (str): Folder name for saving results.
    """
    print(f"🚀 Starting YOLOv8 training on: {data_cfg}")
    print(f"📦 Using model: {model}, epochs={epochs}, imgsz={imgsz}, batch={batch}")

    # Create YOLO model (load pretrained weights)
    yolo = YOLO(model)

    # Train model
    results = yolo.train(
        data=data_cfg,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=name
    )

    print("\n✅ Training complete!")
    print(f"📁 Results saved to: {results.save_dir}")
    print(f"🏆 Best weights: {os.path.join(results.save_dir, 'weights', 'best.pt')}")


# ============================================================
# 🚀 Command-line Interface
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for insect detection.")
    parser.add_argument("--data_cfg", type=str, default="configs/insects.yaml",
                        help="Path to YOLO dataset configuration file.")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model variant or pretrained weight.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (pixels).")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device: 'cpu', 'cuda', or 'mps' (Mac GPU).")
    parser.add_argument("--name", type=str, default="insect_yolo",
                        help="Run name (results folder).")
    args = parser.parse_args()

    train_yolo(
        data_cfg=args.data_cfg,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name
    )

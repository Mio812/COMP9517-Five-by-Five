#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_fastrcnn.py
=========================
Use trained Faster R-CNN model to detect insects in images or directories.

Usage Examples:
--------------- 
# Detect a single image
python detect_fastrcnn.py \
    --weights results/fastrcnn_best.pth \
    --image data/test_images/sample.jpg \
    --class_names beetle fly wasp caterpillar

# Detect all images in a folder
python detect_fastrcnn.py \
    --weights results/fastrcnn_best.pth \
    --image_dir data/test_images \
    --output_dir results/detections \
    --class_names beetle fly wasp caterpillar
"""

import argparse
import torch
import cv2
from pathlib import Path
from torchvision import transforms
from models.faster_rcnn_model import create_faster_rcnn


# ============================================================
# 🧠 Detection Function
# ============================================================
def detect_fastrcnn(weights, image=None, image_dir=None, output_dir="results/detections",
                    conf_thresh=0.5, class_names=None, num_classes=None):
    """
    Run Faster R-CNN inference on single image or folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Create model ---
    if num_classes is None:
        num_classes = len(class_names) + 1  # + background
    model = create_faster_rcnn(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    # --- Prepare paths ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image_dir:
        image_paths = sorted(Path(image_dir).glob("*.jpg"))
    elif image:
        image_paths = [Path(image)]
    else:
        raise ValueError("Please provide either --image or --image_dir.")

    # --- Loop over images ---
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transforms.functional.to_tensor(img_rgb).to(device)

        with torch.no_grad():
            outputs = model([tensor])[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box)
            cls_name = class_names[label - 1] if label > 0 else "background"
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{cls_name} {score:.2f}",
                        (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"✅ Saved detection: {out_path}")

    print("🎉 Detection completed!")


# ============================================================
# 🖥️ Command Line Interface
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Faster R-CNN inference on images.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights (.pth)")
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--image_dir", type=str, help="Path to a folder containing images")
    parser.add_argument("--output_dir", type=str, default="results/detections", help="Where to save detection results")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument(
        "--class_names",
        nargs="+",
        default=[
            "Ants", "Bees", "Beetles", "Caterpillars",
            "Earthworms", "Earwigs", "Grasshoppers",
            "Moths", "Slugs", "Snails", "Wasps", "Weevils"
        ],
        help="List of class names (excluding background)"
    )

    args = parser.parse_args()
    detect_fastrcnn(**vars(args))

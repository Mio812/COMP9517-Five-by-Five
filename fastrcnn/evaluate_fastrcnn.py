
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
from models.faster_rcnn_model import create_faster_rcnn

def load_yolo_labels(label_path, img_w, img_h):
    boxes, labels = [], []
    if not label_path.exists():
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            x1, y1 = (xc - bw / 2) * img_w, (yc - bh / 2) * img_h
            x2, y2 = (xc + bw / 2) * img_w, (yc + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls) + 1)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


def evaluate_fastrcnn_full(weights, data_dir="dataset/test", conf_thresh=0.5, class_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = len(class_names) + 1
    model = create_faster_rcnn(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    img_dir = Path(data_dir) / "images"
    label_dir = Path(data_dir) / "labels"
    image_paths = sorted(img_dir.glob("*.jpg"))

    transform = transforms.Compose([transforms.ToTensor()])
    metric = MeanAveragePrecision(iou_type="bbox")

    all_true, all_pred = [], []

    print(f"Evaluating {len(image_paths)} test images...\n")
    for img_path in tqdm(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        tensor = transform(img_rgb).to(device)
        with torch.no_grad():
            outputs = model([tensor])[0]

        keep = outputs["scores"] > conf_thresh
        boxes_pred = outputs["boxes"][keep].cpu()
        scores_pred = outputs["scores"][keep].cpu()
        labels_pred = outputs["labels"][keep].cpu()

        label_path = label_dir / (img_path.stem + ".txt")
        boxes_true, labels_true = load_yolo_labels(label_path, w, h)

        preds = [{"boxes": boxes_pred, "scores": scores_pred, "labels": labels_pred}]
        targets = [{"boxes": boxes_true, "labels": labels_true}]
        metric.update(preds, targets)

        # Collect for category-level metrics
        all_true.extend(labels_true.numpy().tolist())
        all_pred.extend(labels_pred.numpy().tolist())

    results = metric.compute()
    map50 = results["map_50"].item()
    map5095 = results["map"].item()
    mar100 = results["mar_100"].item()

    min_len = min(len(all_true), len(all_pred))
    all_true = all_true[:min_len]
    all_pred = all_pred[:min_len]

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="weighted", zero_division=0
    )
    acc = accuracy_score(all_true, all_pred)

    print("\nEvaluation Results:")
    print(f"mAP@0.5       : {map50:.4f}")
    print(f"mAP@0.5:0.95  : {map5095:.4f}")
    print(f"mAR@100       : {mar100:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")
    print(f"Accuracy      : {acc:.4f}")

    print("\n Building IoU-matched confusion matrix...")
    cm_true, cm_pred = [], []

    for img_path in tqdm(image_paths, desc="IoU Matching"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")
        boxes_true, labels_true = load_yolo_labels(label_path, w, h)

        tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device)
        with torch.no_grad():
            outputs = model([tensor])[0]

        keep = outputs["scores"] > conf_thresh
        boxes_pred = outputs["boxes"][keep].cpu()
        labels_pred = outputs["labels"][keep].cpu()

        if len(boxes_true) == 0 and len(boxes_pred) == 0:
            continue

        if len(boxes_true) > 0 and len(boxes_pred) > 0:
            ious = box_iou(boxes_true, boxes_pred)
            for i, t_label in enumerate(labels_true):
                best_j = torch.argmax(ious[i])
                best_iou = ious[i, best_j]
                pred_label = labels_pred[best_j]
                if best_iou >= 0.5:
                    cm_true.append(int(t_label))
                    cm_pred.append(int(pred_label))
                else:
                    cm_true.append(int(t_label))
                    cm_pred.append(0)
        else:
            for t in labels_true:
                cm_true.append(int(t))
                cm_pred.append(0)

    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)
    with open(result_dir / "evaluation_results.txt", "w") as f:
        f.write("Faster R-CNN Evaluation Results\n")
        f.write(f"mAP@0.5:      {map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {map5095:.4f}\n")
        f.write(f"mAR@100:      {mar100:.4f}\n")
        f.write(f"Precision:    {precision:.4f}\n")
        f.write(f"Recall:       {recall:.4f}\n")
        f.write(f"F1-score:     {f1:.4f}\n")
        f.write(f"Accuracy:     {acc:.4f}\n")

    cm = confusion_matrix(cm_true, cm_pred, labels=range(1, len(class_names) + 1), normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Value'})
    plt.title("Normalized Confusion Matrix (IoU Matched)", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(result_dir / "confusion_matrix_iou.png", dpi=300)
    plt.close()

    metrics = ["Precision", "Recall", "F1 Score", "Accuracy", "mAP@0.5", "mAP@0.5:0.95"]
    values = [precision, recall, f1, acc, map50, map5095]
    colors = ["#4285F4", "#EA7E23", "#34A853", "#DB4437", "#A142F4", "#9E9E9E"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=colors, edgecolor="black", linewidth=1.5)
    plt.title("Test Set Evaluation Metrics", fontsize=16)
    plt.ylim(0, 1)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=15)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.4f}\n({val*100:.2f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(result_dir / "metrics_bar_chart.png", dpi=300)
    plt.close()

    radar_metrics = ["Precision", "Recall", "F1-score", "Accuracy", "mAP@0.5"]
    radar_values = [precision, recall, f1, acc, map50]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    radar_values += radar_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, radar_values, color="cornflowerblue", linewidth=2)
    ax.fill(angles, radar_values, color="cornflowerblue", alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Metrics Radar Chart", fontsize=14, y=1.1)
    plt.tight_layout()
    plt.savefig(result_dir / "radar_chart.png", dpi=300)
    plt.close()

    print("All evaluation figures saved in results/")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced visualization for Faster R-CNN evaluation (IoU-matched).")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--data_dir", type=str, default="dataset/test", help="Path to dataset folder")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--class_names",
        nargs="+",
        default=[
            "Ants", "Bees", "Beetles", "Caterpillars",
            "Earthworms", "Earwigs", "Grasshoppers", "Moths",
            "Slugs", "Snails", "Wasps", "Weevils",
        ],
        help="List of insect classes (excluding background)"
    )
    args = parser.parse_args()
    evaluate_fastrcnn_full(**vars(args))

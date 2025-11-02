import os
import cv2
import yaml
import time
import torch
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from ultralytics.data.utils import check_det_dataset

def add_gaussian_noise(image, mean=0, var=0.01):
    sigma = var**0.5; row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch)) * 255
    noisy_image = np.clip(image.astype(np.float32) + gauss.reshape(row, col, ch), 0, 255).astype(np.uint8)
    return noisy_image

def apply_blur(image, kernel_size=(7, 7)): return cv2.GaussianBlur(image, kernel_size, 0)
def change_contrast(image, alpha=0.7): return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

DISTORTIONS = {"gaussian_noise": add_gaussian_noise, "blur": apply_blur, "low_contrast": change_contrast}

def load_ground_truth(label_path, img_width, img_height):
    boxes = []
    if not os.path.exists(label_path): return []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            x1, y1 = (x_center - w / 2) * img_width, (y_center - h / 2) * img_height
            x2, y2 = (x_center + w / 2) * img_width, (y_center + h / 2) * img_height
            boxes.append([x1, y1, x2, y2, class_id])
    return boxes

def plot_and_save_comparison(img, gt_boxes, pred_boxes, class_names, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9)); img_gt, img_pred = img.copy(), img.copy()
    for box in gt_boxes:
        x1, y1, x2, y2, class_id = map(int, box)
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_gt, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    ax1.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)); ax1.set_title("Ground Truth"); ax1.axis('off')
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0]); conf, class_id = box.conf[0], int(box.cls[0])
        label = f'{class_names[class_id]} {conf:.2f}'
        cv2.rectangle(img_pred, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_pred, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    ax2.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)); ax2.set_title("Model Prediction"); ax2.axis('off')
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()

def calculate_proxy_accuracy(model, image_paths, label_paths):
    tp, fp, fn, tn = 0, 0, 0, 0
    print("\nCalculating proxy accuracy (this may take some time)...")
    for img_path in tqdm(image_paths, desc="Proxy Accuracy Calculation"):
        label_path = label_paths / (img_path.stem + ".txt")
        has_gt = os.path.exists(label_path) and os.path.getsize(label_path) > 0
        results = model.predict(img_path, verbose=False, device=model.device); preds = results[0].boxes; has_preds = len(preds) > 0
        if not has_gt and not has_preds: tn += 1
        elif has_gt and not has_preds: fn += 1
        elif not has_gt and has_preds: fp += 1
        elif has_gt and has_preds:
            img = cv2.imread(str(img_path)); h, w, _ = img.shape
            gt_boxes_list = load_ground_truth(str(label_path), w, h)
            gt_boxes_tensor = torch.tensor([box[:4] for box in gt_boxes_list]).to(model.device)
            iou_matrix = box_iou(preds.xyxy, gt_boxes_tensor)
            if torch.any(iou_matrix > 0.5): tp += 1
            else: fp += 1; fn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    return {"proxy_accuracy": accuracy}

def plot_per_class_metrics(metrics, class_names, output_path):
    num_classes = len(class_names)
    precision = metrics.box.p
    recall = metrics.box.r
    ap50 = metrics.box.ap50
    
    x = np.arange(num_classes)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(12, num_classes * 0.8), 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#A23B72')
    bars3 = ax.bar(x + width, ap50, width, label='AP50', color='#F18F01')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Per-class performance comparison chart saved: {output_path}")

def plot_robustness_comparison(robustness_results, output_path):
    distortions = list(robustness_results.keys())
    map50_95 = [robustness_results[d]['map50_95'] for d in distortions]
    map50 = [robustness_results[d]['map50'] for d in distortions]
    
    x = np.arange(len(distortions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, map50_95, width, label='mAP50-95', color='#E63946')
    bars2 = ax.bar(x + width/2, map50, width, label='mAP50', color='#457B9D')
    
    ax.set_xlabel('Distortion Type', fontsize=12)
    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title('Robustness Evaluation: Performance Under Different Distortions',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Original'] + [d.replace('_', ' ').title() for d in distortions[1:]])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Robustness comparison chart saved: {output_path}")

def plot_metrics_summary(metrics_dict, output_path):
    metrics_names = ['mAP50-95', 'mAP50', 'Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [
        metrics_dict.get('map50_95', 0),
        metrics_dict.get('map50', 0),
        metrics_dict.get('precision', 0),
        metrics_dict.get('recall', 0),
        metrics_dict.get('f1_score', 0),
        metrics_dict.get('accuracy', 0)
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#E63946', '#F1FAEE', '#A8DADC', '#457B9D', '#1D3557', '#2A9D8F']
    bars = ax.barh(metrics_names, values, color=colors)
    
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Overall Performance Metrics Summary', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.02, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Overall performance chart saved: {output_path}")

def main(args):
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "performance_plots"; plots_dir.mkdir(exist_ok=True)
    
    print(f"Loading model: {args.weights}...")
    model = YOLO(args.weights)

    print("Parsing dataset path using official tools...")
    data_config = check_det_dataset(args.data)
    test_img_path = Path(data_config['test'])
    original_label_path = test_img_path.parent / 'labels'
    class_names = data_config['names']
    print("Path parsing successful!")
    
    if args.run_standard:
        print("\n" + "="*50 + "\nRunning standard performance evaluation\n" + "="*50)
        start_time = time.time()
        
        metrics = model.val(data=args.data, split='test', conf=0.25, iou=0.7,
                           save_json=False, plots=True)
        end_time = time.time()
        
        precision, recall = metrics.box.p[0], metrics.box.r[0]
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        test_image_paths_list = [p for p in test_img_path.glob('*') if p.suffix.lower() in image_extensions]
        num_images = len(test_image_paths_list)

        total_time = end_time - start_time
        inference_time_per_image = total_time / num_images if num_images > 0 else 0

        if num_images > 0 and num_images < 1000:
            accuracy_results = calculate_proxy_accuracy(model, test_image_paths_list, original_label_path)
        else:
            accuracy_results = {"proxy_accuracy": "N/A (Skipped or No Images)"}

        print("\nQuantitative Evaluation Results:")
        print(f"  - mAP50-95: {metrics.box.map:.4f}"); print(f"  - mAP50:    {metrics.box.map50:.4f}")
        print(f"  - Precision: {precision:.4f}"); print(f"  - Recall:    {recall:.4f}")
        print(f"  - F1 Score:  {f1_score:.4f}")
        print(f"  - Proxy Accuracy: {accuracy_results['proxy_accuracy']:.4f}" if isinstance(accuracy_results['proxy_accuracy'], float) else f"  - Proxy Accuracy: {accuracy_results['proxy_accuracy']}")
        print("\nTest Time:"); print(f"  - Total time for {num_images} images: {total_time:.2f} seconds")
        print(f"  - Average inference time per image: {inference_time_per_image*1000:.2f} ms")
        
        print("\nSaving performance visualization charts...")
        yolo_runs_dir = Path("runs/detect")
        if yolo_runs_dir.exists():
            val_dirs = sorted([d for d in yolo_runs_dir.iterdir() if d.is_dir() and d.name.startswith('val')])
            if val_dirs:
                latest_val_dir = val_dirs[-1]
                plot_files = ['confusion_matrix.png', 'F1_curve.png', 'P_curve.png',
                            'R_curve.png', 'PR_curve.png', 'confusion_matrix_normalized.png']
                for plot_file in plot_files:
                    src = latest_val_dir / plot_file
                    if src.exists():
                        shutil.copy2(src, plots_dir / plot_file)
                        print(f"  ✓ Saved: {plot_file}")
        
        print("\nGenerating custom performance charts...")
        
        plot_per_class_metrics(metrics, class_names, plots_dir / "per_class_metrics.png")
        
        metrics_dict = {
            'map50_95': metrics.box.map,
            'map50': metrics.box.map50,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy_results['proxy_accuracy'] if isinstance(accuracy_results['proxy_accuracy'], float) else 0
        }
        plot_metrics_summary(metrics_dict, plots_dir / "overall_metrics_summary.png")
        
        robustness_results = {'original': {'map50_95': metrics.box.map, 'map50': metrics.box.map50}}

    if args.run_robustness:
        print("\n" + "="*50 + "\nRunning robustness evaluation\n" + "="*50)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        source_images = [p for p in test_img_path.glob('*') if p.suffix.lower() in image_extensions]
        
        for name, func in DISTORTIONS.items():
            print(f"\n--- Testing robustness against: {name} ---")
            
            temp_dataset_dir = output_dir / f"temp_dataset_{name}"
            distorted_img_dir = temp_dataset_dir / "images"
            distorted_label_dir = temp_dataset_dir / "labels"
            distorted_img_dir.mkdir(parents=True, exist_ok=True)
            distorted_label_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Generating distorted images and saving to {distorted_img_dir}...")
            for img_file in tqdm(source_images, desc=f"Generating {name} images"):
                img = cv2.imread(str(img_file)); distorted_img = func(img)
                cv2.imwrite(str(distorted_img_dir / img_file.name), distorted_img)

            print(f"Copying original label files to {distorted_label_dir}...")
            for img_file in tqdm(source_images, desc=f"Copying labels"):
                label_name = img_file.stem + ".txt"
                source_label = original_label_path / label_name
                if source_label.exists():
                    shutil.copy2(source_label, distorted_label_dir / label_name)

            temp_config = {
                'path': str(data_config.get('path', '')),
                'train': str(data_config['train']),
                'val': str(distorted_img_dir.resolve()),
                'test': str(distorted_img_dir.resolve()),
                'nc': data_config['nc'],
                'names': data_config['names']
            }
            
            temp_distorted_yaml_path = output_dir / f"temp_distorted_{name}.yaml"
            with open(temp_distorted_yaml_path, 'w') as f:
                yaml.dump(temp_config, f)
            
            metrics_distorted = model.val(data=str(temp_distorted_yaml_path), split='test', conf=0.25, iou=0.7)
            print(f"Results under '{name}' distortion:"); print(f"  - mAP50-95: {metrics_distorted.box.map:.4f} (Original was {metrics.box.map:.4f})")
            
            robustness_results[name] = {
                'map50_95': metrics_distorted.box.map,
                'map50': metrics_distorted.box.map50
            }
        
        print("\nGenerating robustness comparison chart...")
        plot_robustness_comparison(robustness_results, plots_dir / "robustness_comparison.png")

    if args.save_failures or args.save_successes:
        print("\n" + "="*50 + "\nPerforming qualitative analysis\n" + "="*50)
        failures_dir = output_dir / "failure_cases"; successes_dir = output_dir / "success_cases"
        failures_dir.mkdir(exist_ok=True); successes_dir.mkdir(exist_ok=True)
        print("Analyzing prediction results to find success and failure cases...")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = sorted([p for p in test_img_path.glob('*') if p.suffix.lower() in image_extensions])
        for img_path in tqdm(image_files[:args.num_qualitative_images], desc="Qualitative Analysis"):
            img = cv2.imread(str(img_path)); h, w, _ = img.shape
            results = model.predict(img_path, verbose=False, device=model.device); pred_boxes = results[0].boxes
            label_path = original_label_path / (img_path.stem + ".txt")
            gt_boxes_list = load_ground_truth(str(label_path), w, h)
            if not gt_boxes_list: continue
            gt_boxes_tensor = torch.tensor([box[:4] for box in gt_boxes_list]).to(model.device)
            is_failure = False; iou_threshold = 0.5
            if (len(pred_boxes) == 0 and len(gt_boxes_tensor) > 0) or \
               (len(pred_boxes) > 0 and len(gt_boxes_tensor) == 0):
                is_failure = True
            elif len(pred_boxes) > 0 and len(gt_boxes_tensor) > 0:
                iou_matrix = box_iou(pred_boxes.xyxy, gt_boxes_tensor)
                if torch.any(iou_matrix.max(dim=1).values < iou_threshold) or \
                   torch.any(iou_matrix.max(dim=0).values < iou_threshold):
                    is_failure = True
            if is_failure and args.save_failures:
                plot_and_save_comparison(img, gt_boxes_list, pred_boxes, class_names, failures_dir / f"failure_{img_path.name}")
            elif not is_failure and args.save_successes:
                plot_and_save_comparison(img, gt_boxes_list, pred_boxes, class_names, successes_dir / f"success_{img_path.name}")
    
    print("\n" + "="*50)
    print("Evaluation complete! All results have been saved to:", output_dir)
    print(f"  - Performance charts: {plots_dir}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COMP9517 Project - Advanced Evaluation Script for YOLOv8")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained model weights file")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset's YAML configuration file")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Folder to save all evaluation results")
    parser.add_argument("--run-standard", action='store_true', default=True, help="Run standard evaluation")
    parser.add_argument("--run-robustness", action='store_true', help="Run robustness evaluation")
    parser.add_argument("--save-failures", action='store_true', help="Save failure case images")
    parser.add_argument("--save-successes", action='store_true', help="Save success case images")
    parser.add_argument("--num-qualitative-images", type=int, default=30, help="Number of images for qualitative analysis")
    args = parser.parse_args()
    main(args)

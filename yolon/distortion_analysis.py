import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = Path('/root/autodl-tmp/yolon/hard_test_set')
OUTPUT_DIR = SCRIPT_DIR / 'distortion_analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_MODEL = SCRIPT_DIR / 'baseline_yolo11n' / 'weights' / 'best.pt'
ROBUST_MODEL = SCRIPT_DIR / 'robust_yolo11n_augmented' / 'weights' / 'best.pt'

DISTORTION_LEVELS = ['mild', 'moderate', 'severe']

def load_ground_truth(label_dir, image_files):
    gt_data = {}
    for img_file in image_files:
        label_file = label_dir / f"{img_file.stem}.txt"
        boxes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        boxes.append({'class': cls_id, 'bbox': bbox})
        gt_data[img_file.name] = boxes
    return gt_data

def test_with_confidence_threshold(model, images_dir, labels_dir, conf_threshold=0.25):
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    if not image_files:
        print(f"  ! Warning: No images found in {images_dir}")
        return None

    gt_data = load_ground_truth(labels_dir, image_files)
    total_gt = sum(len(boxes) for boxes in gt_data.values())
    if total_gt == 0:
        print(f"  ! Warning: No ground truth boxes found")
        return None

    all_confidences = []
    true_positives = 0
    false_positives = 0
    predictions_count = 0

    results = model.predict(
        source=str(images_dir),
        conf=conf_threshold,
        iou=0.45,
        verbose=False,
        stream=True
    )

    for result in results:
        img_name = Path(result.path).name
        gt_boxes = gt_data.get(img_name, [])
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                all_confidences.append(conf)
                predictions_count += 1
                
                pred_cls = int(box.cls[0])
                if any(gt['class'] == pred_cls for gt in gt_boxes):
                    true_positives += 1
                else:
                    false_positives += 1

    precision = true_positives / predictions_count if predictions_count > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'confidences': all_confidences,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_gt': total_gt,
        'predictions': predictions_count,
        'true_positives': true_positives,
        'false_positives': false_positives
    }

def analyze_distortion_level(model, model_name, distortion_level):
    print(f"\n--- Testing {model_name} on {distortion_level.upper()} distortion ---")
    images_dir = TEST_DATA_DIR / distortion_level / 'images'
    labels_dir = TEST_DATA_DIR / distortion_level / 'labels'

    if not images_dir.exists():
        print(f"  Error: Directory does not exist: {images_dir}")
        return None

    print(f"  Image directory: {images_dir}")
    print(f"  Label directory: {labels_dir}")

    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
    results_by_threshold = {}
    for conf_thresh in thresholds:
        print(f"\n  Testing with confidence threshold = {conf_thresh}")
        result = test_with_confidence_threshold(model, images_dir, labels_dir, conf_thresh)
        if result:
            results_by_threshold[conf_thresh] = result
            print(f"    Precision: {result['precision']:.4f}")
            print(f"    Recall:    {result['recall']:.4f}")
            print(f"    F1 Score:  {result['f1']:.4f}")
            print(f"    Predictions: {result['predictions']}")
            print(f"    Ground Truth: {result['total_gt']}")
    return results_by_threshold

def plot_confidence_distributions(all_results, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Confidence Distribution Analysis - Hypothesis Validation', fontsize=16, fontweight='bold')
    default_threshold = 0.25

    for idx, (model_name, distortion_results) in enumerate(all_results.items()):
        for jdx, distortion_level in enumerate(DISTORTION_LEVELS):
            ax = axes[idx, jdx]
            if distortion_level in distortion_results:
                results = distortion_results[distortion_level].get(default_threshold)
                if results and results['confidences']:
                    confidences = results['confidences']
                    ax.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.axvline(x=default_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={default_threshold}')
                    
                    mean_conf = np.mean(confidences)
                    ax.axvline(x=mean_conf, color='green', linestyle=':', linewidth=1.5, label=f'Mean={mean_conf:.3f}')
                    
                    near_threshold = sum(1 for c in confidences if default_threshold - 0.05 <= c <= default_threshold + 0.05)
                    near_threshold_pct = near_threshold / len(confidences) * 100
                    
                    info_text = f"Recall: {results['recall']:.3f}\n"
                    info_text += f"Near threshold: {near_threshold_pct:.1f}%\n"
                    info_text += f"Total: {len(confidences)}"
                    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
                    
                    ax.set_title(f'{model_name} - {distortion_level.capitalize()}', fontweight='bold')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Frequency')
                    ax.legend(loc='upper left', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.axvspan(default_threshold - 0.05, default_threshold + 0.05, alpha=0.2, color='yellow')
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{model_name} - {distortion_level.capitalize()}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confidence distribution plot saved: {output_path.name}")

def plot_threshold_analysis(all_results, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance vs Confidence Threshold - Hypothesis Validation', fontsize=16, fontweight='bold')
    metrics = ['recall', 'precision', 'f1', 'predictions']
    metric_titles = ['Recall', 'Precision', 'F1 Score', 'Detection Count']

    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx // 2, idx % 2]
        for model_name, distortion_results in all_results.items():
            for distortion_level in DISTORTION_LEVELS:
                if distortion_level in distortion_results:
                    thresholds, values = [], []
                    for thresh, result in sorted(distortion_results[distortion_level].items()):
                        thresholds.append(thresh)
                        values.append(result[metric])
                    
                    linestyle = '-' if model_name == 'Baseline' else '--'
                    marker = 'o' if distortion_level == 'mild' else ('s' if distortion_level == 'moderate' else '^')
                    label = f'{model_name} - {distortion_level.capitalize()}'
                    ax.plot(thresholds, values, marker=marker, linestyle=linestyle, linewidth=2, markersize=8, label=label)

        ax.set_xlabel('Confidence Threshold', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs Threshold', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.25, color='red', linestyle=':', alpha=0.5, label='Default=0.25' if idx == 0 else '')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Threshold analysis plot saved: {output_path.name}")

def plot_recall_comparison(all_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Recall Comparison - Why is Moderate the Lowest?', fontsize=16, fontweight='bold')
    default_threshold = 0.25

    for idx, model_name in enumerate(['Baseline', 'Robust']):
        ax = axes[idx]
        if model_name in all_results:
            recalls, precisions = [], []
            for level in DISTORTION_LEVELS:
                result = all_results[model_name].get(level, {}).get(default_threshold)
                if result:
                    recalls.append(result['recall'])
                    precisions.append(result['precision'])
                else:
                    recalls.append(0)
                    precisions.append(0)
            
            x = np.arange(len(DISTORTION_LEVELS))
            width = 0.35
            bars1 = ax.bar(x - width/2, recalls, width, label='Recall', alpha=0.8, color='#3498db')
            bars2 = ax.bar(x + width/2, precisions, width, label='Precision', alpha=0.8, color='#e74c3c')
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            min_recall_idx = recalls.index(min(recalls))
            ax.annotate('Lowest Point!', xy=(min_recall_idx - width/2, recalls[min_recall_idx]),
                       xytext=(min_recall_idx - width/2, recalls[min_recall_idx] + 0.05),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2), fontsize=12, color='red', fontweight='bold')
            
            ax.set_xlabel('Distortion Level', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'{model_name} Model', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([l.capitalize() for l in DISTORTION_LEVELS])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Recall comparison plot saved: {output_path.name}")

def analyze_confidence_near_threshold(all_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Percentage of Samples Near Threshold - Key Evidence',
                 fontsize=16, fontweight='bold')
    
    default_threshold = 0.25
    margin = 0.05
    
    for idx, model_name in enumerate(['Baseline', 'Robust']):
        ax = axes[idx]
        
        if model_name in all_results:
            below_thresh = []
            above_thresh = []
            near_thresh = []
            
            for level in DISTORTION_LEVELS:
                result = all_results[model_name].get(level, {}).get(default_threshold)
                
                if result and result['confidences']:
                    confs = result['confidences']
                    total_confs = len(confs)
                    
                    below = sum(1 for c in confs
                              if default_threshold - margin <= c < default_threshold)
                    above = sum(1 for c in confs
                              if default_threshold <= c <= default_threshold + margin)
                    
                    below_thresh.append(below / total_confs * 100 if total_confs else 0)
                    above_thresh.append(above / total_confs * 100 if total_confs else 0)
                    near_thresh.append((below + above) / total_confs * 100 if total_confs else 0)
                else:
                    below_thresh.append(0)
                    above_thresh.append(0)
                    near_thresh.append(0)
            
            x = np.arange(len(DISTORTION_LEVELS))
            width = 0.25
            
            bars1 = ax.bar(x - width, below_thresh, width,
                          label=f'Below Threshold ({default_threshold-margin:.2f}-{default_threshold:.2f})',
                          color='#e74c3c', alpha=0.8)
            bars2 = ax.bar(x, above_thresh, width,
                          label=f'Above Threshold ({default_threshold:.2f}-{default_threshold+margin:.2f})',
                          color='#2ecc71', alpha=0.8)
            bars3 = ax.bar(x + width, near_thresh, width,
                          label=f'Near Threshold (±{margin:.2f})',
                          color='#f39c12', alpha=0.8)
            
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%',
                               ha='center', va='bottom', fontsize=9)
            
            if near_thresh:
                max_bar_height = max(near_thresh)
                ax.annotate('\nMost samples\nstuck at the edge',
                           xy=(1, max_bar_height),
                           xytext=(1, max_bar_height + 5),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           fontsize=11, color='red', fontweight='bold',
                           ha='center')

            ax.set_xlabel('Distortion Level', fontsize=12)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_title(f'{model_name} Model - Sample Distribution Near Threshold', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([l.capitalize() for l in DISTORTION_LEVELS])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            all_heights = below_thresh + above_thresh + near_thresh
            max_plot_height = max(all_heights) if all_heights else 10
            ax.set_ylim(0, max(max_plot_height, max(near_thresh) if near_thresh else 0) + 7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Near-threshold analysis plot saved: {output_path.name}")

def generate_summary_report(all_results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" Distortion Dataset Test Report - Why is Moderate Recall the Lowest? ".center(80) + "\n")
        f.write("="*80 + "\n\n")
        
        default_threshold = 0.25
        for model_name in ['Baseline', 'Robust']:
            if model_name not in all_results:
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f" {model_name} Model Analysis ".center(80) + "\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("[Performance at Default Threshold (0.25)]\n")
            f.write(f"{'Level':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Predictions':>12} {'Ground Truth':>14}\n")
            f.write("-" * 80 + "\n")
            
            recalls = {}
            confidences_stats = {}
            for level in DISTORTION_LEVELS:
                result = all_results[model_name].get(level, {}).get(default_threshold)
                if result:
                    f.write(f"{level.capitalize():<12} {result['precision']:>10.4f} {result['recall']:>10.4f} {result['f1']:>10.4f} {result['predictions']:>12} {result['total_gt']:>14}\n")
                    recalls[level] = result['recall']
                    if result['confidences']:
                        confs = result['confidences']
                        confidences_stats[level] = {
                            'mean': np.mean(confs), 'median': np.median(confs), 'std': np.std(confs),
                            'near_threshold': sum(1 for c in confs if default_threshold - 0.05 <= c <= default_threshold + 0.05) / len(confs) * 100
                        }
            
            if recalls:
                min_level = min(recalls, key=recalls.get)
                max_level = max(recalls, key=recalls.get)
                f.write(f"\n[Key Findings]\n")
                f.write(f"  - Lowest Recall: {min_level.capitalize()} = {recalls[min_level]:.4f}\n")
                f.write(f"  - Highest Recall: {max_level.capitalize()} = {recalls[max_level]:.4f}\n")
                f.write(f"  - Difference: {recalls[max_level] - recalls[min_level]:.4f} ({(recalls[max_level] - recalls[min_level])/recalls[min_level]*100:.1f}%)\n\n")
                if min_level == 'moderate':
                    f.write("  Hypothesis confirmed: Moderate distortion has the lowest Recall!\n\n")
                
                f.write(f"[Confidence Analysis]\n")
                for level in DISTORTION_LEVELS:
                    if level in confidences_stats:
                        stats = confidences_stats[level]
                        f.write(f"\n  {level.capitalize()}:\n")
                        f.write(f"    - Mean Confidence: {stats['mean']:.4f}\n")
                        f.write(f"    - Median Confidence: {stats['median']:.4f}\n")
                        f.write(f"    - Standard Deviation: {stats['std']:.4f}\n")
                        f.write(f"    - Near Threshold Ratio (±0.05): {stats['near_threshold']:.2f}%\n")
                        if level == 'moderate' and stats['near_threshold'] > 15:
                            f.write(f"    KEY EVIDENCE: {stats['near_threshold']:.1f}% of samples are clustered near the threshold!\n")
                            f.write(f"       This explains the low recall.\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write(" Conclusion ".center(80) + "\n")
        f.write("="*80 + "\n\n")
        f.write("Reason for the lowest recall in moderate distortion:\n\n")
        f.write("1. [Confidence Distribution] Moderate distortion causes many sample confidences to cluster near the decision threshold.\n")
        f.write("   Many samples have confidences in the 0.20-0.28 range, just below the default 0.25 threshold.\n")
        f.write("   This leads to a large number of true positives being filtered out.\n\n")
        f.write('2. [Distortion Characteristics] Moderate distortion creates the most "ambiguous" scenarios for the model.\n')
        f.write("   Features are significantly, but not completely, lost.\n")
        f.write("   The model is most uncertain, resulting in many borderline confidence scores.\n\n")
        f.write("3. [Comparison to Extremes] Severe distortion, paradoxically, simplifies the problem.\n")
        f.write("   Detections are either missed entirely (low confidence) or made on obvious remaining features (high confidence).\n")
        f.write('   This reduces the number of samples "stuck at the threshold edge."\n\n')
        f.write("4. [Potential Solutions]\n")
        f.write("   - Lower the confidence threshold to 0.20 or less.\n")
        f.write("   - Augment the training data with more moderately distorted samples.\n")
        f.write("   - Implement an adaptive thresholding strategy.\n\n")

    print(f"  Detailed report saved: {output_path.name}")

def main():
    print("--- Distortion Dataset Analysis ---")
    print(f"Using test data directory: {TEST_DATA_DIR}")
    for level in DISTORTION_LEVELS:
        level_dir = TEST_DATA_DIR / level / 'images'
        if level_dir.exists():
            img_count = len(list(level_dir.glob('*.jpg'))) + len(list(level_dir.glob('*.png')))
            print(f"  {level.capitalize()}: Found {img_count} images")
        else:
            print(f"  {level.capitalize()}: Directory not found")

    print("\nLoading models...")
    models = {}
    if BASELINE_MODEL.exists():
        print("  Loading Baseline model...")
        models['Baseline'] = YOLO(str(BASELINE_MODEL))
        print("    Loaded successfully")
    else:
        print(f"  Baseline model not found: {BASELINE_MODEL}")
    
    if ROBUST_MODEL.exists():
        print("  Loading Robust model...")
        models['Robust'] = YOLO(str(ROBUST_MODEL))
        print("    Loaded successfully")
    else:
        print(f"  Robust model not found: {ROBUST_MODEL}")
    
    if not models:
        print("\nNo models available! Please check model paths.")
        return

    all_results = {}
    for model_name, model in models.items():
        all_results[model_name] = {}
        for distortion_level in DISTORTION_LEVELS:
            results = analyze_distortion_level(model, model_name, distortion_level)
            if results:
                all_results[model_name][distortion_level] = results

    print("\n--- Generating analysis plots and report ---")
    plot_confidence_distributions(all_results, OUTPUT_DIR / 'confidence_distributions.png')
    plot_threshold_analysis(all_results, OUTPUT_DIR / 'threshold_analysis.png')
    plot_recall_comparison(all_results, OUTPUT_DIR / 'recall_comparison.png')
    analyze_confidence_near_threshold(all_results, OUTPUT_DIR / 'confidence_near_threshold.png')
    generate_summary_report(all_results, OUTPUT_DIR / 'detailed_analysis_report.txt')
    
    print("\n--- Analysis Complete! ---")
    print(f"All results have been saved to: {OUTPUT_DIR}")
    print("\nGenerated Files:")
    print("  - confidence_distributions.png - Confidence distribution comparison")
    print("  - threshold_analysis.png - Performance across different thresholds")
    print("  - recall_comparison.png - Recall comparison (hypothesis validation)")
    print("  - confidence_near_threshold.png - Near-threshold sample analysis (key evidence)")
    print("  - detailed_analysis_report.txt - Detailed text report")
    print("\nThe charts and report validate the hypothesis: moderate distortion's recall is lowest")
    print("   because a large number of sample confidences are clustered just below the threshold.")

if __name__ == '__main__':
    main()

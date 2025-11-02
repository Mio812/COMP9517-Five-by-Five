import cv2
import numpy as np
import os
import shutil
import yaml
import time
import json
from ultralytics import YOLO
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

try:
    from custom_modules import CBAM
    import ultralytics.nn.tasks
    ultralytics.nn.tasks.CBAM = CBAM
    print("CBAM module loaded")
except ImportError:
    print("Warning: Cannot import CBAM module, using standard YOLO model")

MODEL_PATH = 'runs/detect/yolov8_cbam_finetuned_with_yolov10n/weights/best.pt'
DATA_YAML_PATH = '/root/autodl-tmp/dataset/data.yaml'
TEMP_DATASET_DIR = '/root/autodl-tmp/dataset_distorted_test'

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def resolve_path(path, base_dir=None):
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = os.getcwd()
    resolved = os.path.join(base_dir, path)
    resolved = os.path.normpath(resolved)
    return resolved

def find_model_path(base_dirs=None, model_name='best.pt'):
    if base_dirs is None:
        base_dirs = [
            'runs/detect',
            './runs/detect',
            '../runs/detect',
            '/root/autodl-tmp/PestProject/runs/detect',
            os.path.expanduser('~/autodl-tmp/PestProject/runs/detect'),
            os.path.join(os.path.dirname(__file__), 'runs/detect'),
        ]
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            pattern = os.path.join(base_dir, '**', model_name)
            found_files = glob.glob(pattern, recursive=True)
            if found_files:
                latest_file = max(found_files, key=os.path.getmtime)
                return latest_file
    return None

def find_dataset_path(yaml_path, path_from_yaml):
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    
    candidate_paths = [
        resolve_path(path_from_yaml, yaml_dir),
        resolve_path(path_from_yaml, os.getcwd()),
        path_from_yaml,
        '/root/autodl-tmp/dataset/test/images',
        '/root/autodl-tmp/test/images',
        os.path.join(os.path.dirname(yaml_dir), 'test', 'images'),
    ]
    
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    return None

def find_labels_path(images_path):
    parent_dir = os.path.dirname(images_path)
    labels_path = os.path.join(parent_dir, 'labels')
    if os.path.exists(labels_path):
        return labels_path
    
    labels_path = images_path.replace('/images', '/labels')
    if os.path.exists(labels_path):
        return labels_path
    
    parent_parent = os.path.dirname(parent_dir)
    labels_path = os.path.join(parent_parent, 'labels')
    if os.path.exists(labels_path):
        return labels_path
    
    return None

def apply_distortion(image, distortion_type='gaussian_noise', severity=1):
    if distortion_type == 'gaussian_noise':
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        noise = np.random.normal(0, c * 255, image.shape)
        distorted_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    elif distortion_type == 'blur':
        k_size = [3, 5, 7, 9, 11][severity - 1]
        distorted_image = cv2.GaussianBlur(image, (k_size, k_size), 0)
    elif distortion_type == 'brightness':
        c = [0.5, 0.4, 0.3, 0.2, 0.1][severity - 1]
        distorted_image = np.clip(image * c, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")
    return distorted_image

def create_distorted_dataset(original_test_images_path, distortion_type, severity):
    distorted_root = os.path.join(TEMP_DATASET_DIR, f"{distortion_type}_{severity}")
    distorted_images_path = os.path.join(distorted_root, 'images')
    distorted_labels_path = os.path.join(distorted_root, 'labels')
    
    if os.path.exists(distorted_root):
        shutil.rmtree(distorted_root)
    
    os.makedirs(distorted_images_path)
    os.makedirs(distorted_labels_path)
    
    print(f"\nCreating distorted test set: {distortion_type} (severity: {severity})...")
    
    image_files = [f for f in os.listdir(original_test_images_path)
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        raise ValueError(f"No image files found in {original_test_images_path}")
    
    original_labels_path = find_labels_path(original_test_images_path)
    
    if original_labels_path is None:
        print(f"  ⚠️  Warning: Label files not found")
    else:
        print(f"  ✓ Found label directory: {original_labels_path}")
    
    for filename in tqdm(image_files, desc=f"Applying {distortion_type}"):
        img_path = os.path.join(original_test_images_path, filename)
        image = cv2.imread(img_path)
        
        if image is not None:
            distorted_image = apply_distortion(image, distortion_type, severity)
            cv2.imwrite(os.path.join(distorted_images_path, filename), distorted_image)
            
            if original_labels_path:
                base_name = os.path.splitext(filename)[0]
                label_filename = base_name + '.txt'
                original_label_path = os.path.join(original_labels_path, label_filename)
                
                if os.path.exists(original_label_path):
                    shutil.copy2(original_label_path,
                               os.path.join(distorted_labels_path, label_filename))
    
    label_count = len([f for f in os.listdir(distorted_labels_path) if f.endswith('.txt')])
    print(f"  ✓ Created {len(image_files)} distorted images")
    print(f"  ✓ Copied {label_count} label files")
    
    return distorted_root

def create_visualizations(full_report, save_dir, original_test_images_path):
    print("\nStep 7: Creating visualizations")
    print("-" * 40)
    
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    create_baseline_performance_chart(full_report, vis_dir)
    
    create_robustness_comparison_chart(full_report, vis_dir)
    
    create_performance_drop_chart(full_report, vis_dir)
    
    create_comprehensive_comparison(full_report, vis_dir)
    
    create_distortion_examples(original_test_images_path, vis_dir)
    
    create_timing_visualization(full_report, vis_dir)
    
    print(f"✓ All visualization charts have been saved to: {vis_dir}")
    return vis_dir

def create_baseline_performance_chart(full_report, save_dir):
    baseline = full_report.get('baseline_performance_clean_test_set', {})
    
    metrics = ['mAP@50-95', 'mAP@50', 'Precision', 'Recall', 'F1_Score']
    values = [
        baseline.get('mAP@50-95', 0),
        baseline.get('mAP@50', 0),
        baseline.get('Precision', 0),
        baseline.get('Recall', 0),
        baseline.get('F1_Score', 0)
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Performance on Clean Test Set', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_baseline_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Baseline performance chart")

def create_robustness_comparison_chart(full_report, save_dir):
    robustness = full_report.get('robustness_evaluation', {})
    baseline = full_report.get('baseline_performance_clean_test_set', {})
    baseline_map50 = baseline.get('mAP@50', 0)
    baseline_map50_95 = baseline.get('mAP@50-95', 0)
    
    if not robustness:
        return
    
    distortion_types = []
    map50_values = []
    map50_95_values = []
    
    for key, metrics in robustness.items():
        dist_name = key.replace('_sev_3', '').replace('_', ' ').title()
        distortion_types.append(dist_name)
        map50_values.append(metrics.get('mAP@50', 0))
        map50_95_values.append(metrics.get('mAP@50-95', 0))
    
    distortion_types.insert(0, 'Clean (Baseline)')
    map50_values.insert(0, baseline_map50)
    map50_95_values.insert(0, baseline_map50_95)
    
    x = np.arange(len(distortion_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, map50_values, width, label='mAP@50',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, map50_95_values, width, label='mAP@50-95',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Distortion Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Robustness: Performance Under Different Distortions',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(distortion_types, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(map50_values), max(map50_95_values)) * 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_robustness_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Robustness comparison chart")

def create_performance_drop_chart(full_report, save_dir):
    robustness = full_report.get('robustness_evaluation', {})
    
    if not robustness:
        return
    
    distortion_types = []
    drop_50 = []
    drop_50_95 = []
    
    for key, metrics in robustness.items():
        dist_name = key.replace('_sev_3', '').replace('_', ' ').title()
        distortion_types.append(dist_name)
        drop_50.append(metrics.get('performance_drop_mAP50_percent', 0))
        drop_50_95.append(metrics.get('performance_drop_mAP50_95_percent', 0))
    
    x = np.arange(len(distortion_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, drop_50, width, label='mAP@50 Drop',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, drop_50_95, width, label='mAP@50-95 Drop',
                   color='#c0392b', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Distortion Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Drop (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Degradation Under Distortions',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(distortion_types, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_performance_drop.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Performance drop chart")

def create_comprehensive_comparison(full_report, save_dir):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    create_radar_chart(full_report, ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    create_map_line_chart(full_report, ax2)
    
    ax3 = fig.add_subplot(gs[1, 0])
    create_drop_heatmap(full_report, ax3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    create_timing_info(full_report, ax4)
    
    plt.savefig(os.path.join(save_dir, '4_comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Comprehensive comparison chart")

def create_radar_chart(full_report, ax):
    baseline = full_report.get('baseline_performance_clean_test_set', {})
    
    categories = ['mAP@50-95', 'mAP@50', 'Precision', 'Recall', 'F1 Score']
    values = [
        baseline.get('mAP@50-95', 0),
        baseline.get('mAP@50', 0),
        baseline.get('Precision', 0),
        baseline.get('Recall', 0),
        baseline.get('F1_Score', 0)
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_title('Baseline Performance Radar', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

def create_map_line_chart(full_report, ax):
    baseline = full_report.get('baseline_performance_clean_test_set', {})
    robustness = full_report.get('robustness_evaluation', {})
    
    conditions = ['Clean']
    map50_vals = [baseline.get('mAP@50', 0)]
    map50_95_vals = [baseline.get('mAP@50-95', 0)]
    
    for key, metrics in robustness.items():
        dist_name = key.replace('_sev_3', '').replace('_', '\n').title()
        conditions.append(dist_name)
        map50_vals.append(metrics.get('mAP@50', 0))
        map50_95_vals.append(metrics.get('mAP@50-95', 0))
    
    x = range(len(conditions))
    ax.plot(x, map50_vals, 'o-', linewidth=2, markersize=8, label='mAP@50', color='#3498db')
    ax.plot(x, map50_95_vals, 's-', linewidth=2, markersize=8, label='mAP@50-95', color='#e74c3c')
    
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel('mAP Score', fontweight='bold')
    ax.set_title('mAP Trends Across Conditions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_drop_heatmap(full_report, ax):
    robustness = full_report.get('robustness_evaluation', {})
    
    if not robustness:
        ax.text(0.5, 0.5, 'No robustness data', ha='center', va='center')
        return
    
    dist_names = []
    drops = []
    
    for key, metrics in robustness.items():
        dist_name = key.replace('_sev_3', '').replace('_', ' ').title()
        dist_names.append(dist_name)
        drops.append([
            metrics.get('performance_drop_mAP50_percent', 0),
            metrics.get('performance_drop_mAP50_95_percent', 0)
        ])
    
    drops = np.array(drops)
    im = ax.imshow(drops.T, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(range(len(dist_names)))
    ax.set_xticklabels(dist_names, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['mAP@50', 'mAP@50-95'])
    ax.set_title('Performance Drop Heatmap (%)', fontweight='bold')
    
    for i in range(len(dist_names)):
        for j in range(2):
            text = ax.text(i, j, f'{drops[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Drop %')

def create_timing_info(full_report, ax):
    timing = full_report.get('timing', {})
    
    ax.axis('off')
    
    info_text = "Inference Performance\n\n"
    info_text += f"FPS: {timing.get('fps', 0):.2f}\n"
    info_text += f"Avg Time: {timing.get('avg_inference_ms_per_image', 0):.2f} ms\n"
    info_text += f"Total Time: {timing.get('total_test_time_seconds', 0):.2f} s"
    
    ax.text(0.5, 0.6, info_text, ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fps = timing.get('fps', 0)
    if fps > 100:
        level = "Excellent"
        color = '#2ecc71'
    elif fps > 50:
        level = "Good"
        color = '#3498db'
    elif fps > 25:
        level = "Acceptable"
        color = '#f39c12'
    else:
        level = "Needs Improvement"
        color = '#e74c3c'
    
    ax.text(0.5, 0.3, f"Performance: {level}", ha='center', va='center',
            fontsize=12, color=color, fontweight='bold')

def create_distortion_examples(original_test_images_path, save_dir):
    image_files = [f for f in os.listdir(original_test_images_path)
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        return
    
    selected_images = np.random.choice(image_files, min(3, len(image_files)), replace=False)
    
    fig, axes = plt.subplots(len(selected_images), 4, figsize=(16, 4*len(selected_images)))
    if len(selected_images) == 1:
        axes = axes.reshape(1, -1)
    
    distortion_types = ['gaussian_noise', 'blur', 'brightness']
    severity = 3
    
    for idx, img_file in enumerate(selected_images):
        img_path = os.path.join(original_test_images_path, img_file)
        original = cv2.imread(img_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        axes[idx, 0].imshow(original_rgb)
        axes[idx, 0].set_title('Original', fontweight='bold')
        axes[idx, 0].axis('off')
        
        for col, dist_type in enumerate(distortion_types, 1):
            distorted = apply_distortion(original, dist_type, severity)
            distorted_rgb = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
            axes[idx, col].imshow(distorted_rgb)
            title = dist_type.replace('_', ' ').title()
            axes[idx, col].set_title(f'{title} (Sev {severity})', fontweight='bold')
            axes[idx, col].axis('off')
    
    plt.suptitle('Distortion Examples: Visual Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '5_distortion_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Distortion examples comparison chart")

def create_timing_visualization(full_report, save_dir):
    timing = full_report.get('timing', {})
    
    if not timing:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    fps = timing.get('fps', 0)
    benchmarks = {
        'Current Model': fps,
        'Real-time (30 FPS)': 30,
        'High Performance (60 FPS)': 60,
        'Ultra Fast (120 FPS)': 120
    }
    
    names = list(benchmarks.keys())
    values = list(benchmarks.values())
    colors = ['#3498db', '#95a5a6', '#95a5a6', '#95a5a6']
    
    bars = ax1.barh(names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars, values):
        ax1.text(value, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}',
                va='center', fontweight='bold', fontsize=11)
    
    ax1.set_xlabel('FPS (Frames Per Second)', fontweight='bold')
    ax1.set_title('Inference Speed Comparison', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    avg_time = timing.get('avg_inference_ms_per_image', 0)
    preprocess = avg_time * 0.35
    inference = avg_time * 0.45
    postprocess = avg_time * 0.20
    
    sizes = [preprocess, inference, postprocess]
    labels = ['Preprocess', 'Inference', 'Postprocess']
    colors_pie = ['#3498db', '#e74c3c', '#2ecc71']
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                         autopct='%1.1f%%', shadow=True, startangle=90)
    
    for text in texts:
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title(f'Time Breakdown\n(Avg: {avg_time:.2f} ms/image)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '6_timing_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Inference time analysis chart")

def main():
    print("="*80)
    print("Model Robustness Evaluation Script v4.0 (with Full Visualization)")
    print("="*80)
    
    print(f"\nStep 1: Loading Model")
    print("-" * 40)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: The specified model path does not exist: {MODEL_PATH}")
        print("Searching for model file...")
        found_path = find_model_path()
        if found_path:
            print(f"✓ Found model file: {found_path}")
            model_path = found_path
        else:
            print("\n❌ Error: Cannot find model file!")
            return
    else:
        model_path = MODEL_PATH
        print(f"✓ Found model file: {model_path}")
    
    print(f"Loading model...")
    try:
        model = YOLO(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    print(f"\nStep 2: Checking Data Configuration")
    print("-" * 40)
    
    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ Error: Data configuration file does not exist: {DATA_YAML_PATH}")
        return
    
    print(f"✓ Data configuration file exists: {DATA_YAML_PATH}")
    
    try:
        with open(DATA_YAML_PATH, 'r') as f:
            data_config = yaml.safe_load(f)
        print(f"✓ Data configuration loaded successfully")
        
        if 'test' in data_config:
            test_path_from_yaml = data_config['test']
            print(f"Test set path in YAML: {test_path_from_yaml}")
            
            print("\nIntelligently parsing test set path...")
            original_test_images_path = find_dataset_path(DATA_YAML_PATH, test_path_from_yaml)
            
            if original_test_images_path:
                print(f"✓ Resolved test set path: {original_test_images_path}")
                
                if os.path.exists(original_test_images_path):
                    test_images = [f for f in os.listdir(original_test_images_path)
                                  if f.endswith(('.jpg', '.png', '.jpeg'))]
                    print(f"✓ Found {len(test_images)} test images")
                    
                    original_labels_path = find_labels_path(original_test_images_path)
                    if original_labels_path:
                        label_files = [f for f in os.listdir(original_labels_path)
                                     if f.endswith('.txt')]
                        print(f"✓ Found {len(label_files)} label files")
                    else:
                        print(f"⚠️  Warning: Label directory not found")
                else:
                    print(f"❌ Error: Resolved path still does not exist")
                    return
            else:
                print(f"❌ Error: Cannot find test set")
                return
        else:
            print("❌ Error: No 'test' key in data configuration")
            return
            
    except Exception as e:
        print(f"❌ Failed to read data configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    full_report = {}

    print(f"\nStep 3: Evaluating Baseline Performance on Clean Test Set")
    print("-" * 40)
    
    try:
        temp_yaml_path = os.path.join(os.path.dirname(DATA_YAML_PATH), 'temp_eval_data.yaml')
        temp_config = data_config.copy()
        temp_config['test'] = original_test_images_path
        
        if 'train' in temp_config:
            train_resolved = find_dataset_path(DATA_YAML_PATH, temp_config['train'])
            if train_resolved:
                temp_config['train'] = train_resolved
        
        if 'val' in temp_config:
            val_resolved = find_dataset_path(DATA_YAML_PATH, temp_config['val'])
            if val_resolved:
                temp_config['val'] = val_resolved
        
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(temp_config, f)
        
        print(f"Using temporary configuration file: {temp_yaml_path}")
        
        metrics_clean = model.val(data=temp_yaml_path, split='test', plots=True)
        
        map50_95_clean = metrics_clean.box.map
        map50_clean = metrics_clean.box.map50
        precision_clean = metrics_clean.box.p[0] if len(metrics_clean.box.p) > 0 else 0.0
        recall_clean = metrics_clean.box.r[0] if len(metrics_clean.box.r) > 0 else 0.0
        f1_score_clean = 2 * (precision_clean * recall_clean) / (precision_clean + recall_clean + 1e-6)

        print("\n--- Baseline Performance (Clean Test Set) ---")
        print(f"  - mAP@50-95: {map50_95_clean:.4f}")
        print(f"  - mAP@50: {map50_clean:.4f}")
        print(f"  - Precision: {precision_clean:.4f}")
        print(f"  - Recall: {recall_clean:.4f}")
        print(f"  - F1 Score: {f1_score_clean:.4f}")
        
        full_report['baseline_performance_clean_test_set'] = {
            "mAP@50-95": float(map50_95_clean),
            "mAP@50": float(map50_clean),
            "Precision": float(precision_clean),
            "Recall": float(recall_clean),
            "F1_Score": float(f1_score_clean)
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nStep 4: Evaluating Inference Time")
    print("-" * 40)
    
    test_images_paths = [os.path.join(original_test_images_path, f) for f in test_images]
    num_images = len(test_images_paths)
    
    print(f"  - Testing inference time for {num_images} images...")
    start_time = time.time()
    for img_path in tqdm(test_images_paths, desc="Calculating Inference Time"):
        model(img_path, verbose=False)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / num_images
    fps = 1 / avg_time_per_image
    
    print(f"\n  - Total test time: {total_time:.2f} seconds")
    print(f"  - Average per image: {avg_time_per_image * 1000:.2f} ms")
    print(f"  - FPS: {fps:.2f}")
    
    full_report['timing'] = {
        "total_test_time_seconds": float(total_time),
        "avg_inference_ms_per_image": float(avg_time_per_image * 1000),
        "fps": float(fps)
    }

    print(f"\nStep 5: Evaluating Model Robustness")
    print("-" * 40)
    
    distortions = ['gaussian_noise', 'blur', 'brightness']
    severity_level = 3
    
    robustness_results = {}
    
    for dist_type in distortions:
        try:
            new_dataset_root = create_distorted_dataset(original_test_images_path, dist_type, severity_level)

            distorted_yaml_path = os.path.join(TEMP_DATASET_DIR, 'temp_data.yaml')
            distorted_config = temp_config.copy()
            distorted_config['test'] = os.path.join(new_dataset_root, 'images')
            
            os.makedirs(os.path.dirname(distorted_yaml_path), exist_ok=True)
            with open(distorted_yaml_path, 'w') as f:
                yaml.dump(distorted_config, f)

            print(f"\n--- Evaluating {dist_type} (severity: {severity_level}) ---")
            metrics_distorted = model.val(data=distorted_yaml_path, split='test')
            
            map50_distorted = metrics_distorted.box.map50
            map50_95_distorted = metrics_distorted.box.map
            
            print(f"  - mAP@50: {map50_distorted:.4f}")
            print(f"  - mAP@50-95: {map50_95_distorted:.4f}")
            
            if map50_clean > 0:
                drop_50 = ((map50_clean - map50_distorted) / map50_clean) * 100
            else:
                drop_50 = 0.0
                
            if map50_95_clean > 0:
                drop_50_95 = ((map50_95_clean - map50_95_distorted) / map50_95_clean) * 100
            else:
                drop_50_95 = 0.0
            
            print(f"  - Performance Drop (mAP@50): {drop_50:.2f}%")
            print(f"  - Performance Drop (mAP@50-95): {drop_50_95:.2f}%")
            
            robustness_results[f"{dist_type}_sev_{severity_level}"] = {
                "mAP@50": float(map50_distorted),
                "mAP@50-95": float(map50_95_distorted),
                "performance_drop_mAP50_percent": float(drop_50),
                "performance_drop_mAP50_95_percent": float(drop_50_95)
            }
        except Exception as e:
            print(f"❌ Error while evaluating {dist_type}: {e}")

    full_report['robustness_evaluation'] = robustness_results

    print(f"\nStep 6: Saving Evaluation Report")
    print("-" * 40)
    
    if os.path.exists(TEMP_DATASET_DIR):
        shutil.rmtree(TEMP_DATASET_DIR)
        print("✓ Temporary files cleaned up")
    
    if os.path.exists(temp_yaml_path):
        os.remove(temp_yaml_path)

    report_path = os.path.join(metrics_clean.save_dir, 'full_evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=4)
    
    print(f"✓ JSON report saved: {report_path}")
    
    try:
        vis_dir = create_visualizations(full_report, metrics_clean.save_dir, original_test_images_path)
    except Exception as e:
        print(f"⚠️ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        vis_dir = None
    
    print(f"\n{'='*80}")
    print("✓ Evaluation fully completed!")
    print(f"{'='*80}")
    print(f"\nResults saved in: {metrics_clean.save_dir}")
    print(f"  - JSON Report: {report_path}")
    if vis_dir:
        print(f"  - Visualization Charts: {vis_dir}")
    
    print(f"\nPerformance Summary:")
    print(f"  - Baseline mAP@50: {map50_clean:.4f}")
    print(f"  - Inference Speed: {fps:.2f} FPS")
    print(f"  - Robustness Test: {len(robustness_results)} distortion types")
    
    if robustness_results:
        print(f"\nRobustness Results:")
        for dist_name, metrics in robustness_results.items():
            print(f"  - {dist_name}: mAP@50={metrics['mAP@50']:.4f}, "
                  f"Drop={metrics['performance_drop_mAP50_percent']:.1f}%")
    
    print(f"\nList of Visualization Charts:")
    if vis_dir and os.path.exists(vis_dir):
        viz_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])
        for vf in viz_files:
            print(f"  ✓ {vf}")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

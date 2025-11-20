# Agricultural Pest Detection Project

## Project Overview

This project is a comprehensive computer vision system for agricultural pest detection and classification. It implements multiple state-of-the-art deep learning and traditional machine learning approaches to automatically identify and localize various agricultural pests. The project includes extensive evaluation metrics, robustness testing, and visualization tools to compare different detection methods.
Full project address


### Key Features

- **Multiple Detection Models**: YOLO11n, Faster R-CNN, and SVM-based classification
- **Attention Mechanisms**: CBAM (Convolutional Block Attention Module) integration
- **Comprehensive Evaluation**: mAP, Precision, Recall, F1-Score, AUC, and more
- **Data Augmentation**: Advanced augmentation strategies for improved robustness
- **Rich Visualizations**: Training curves, confusion matrices, PR/ROC curves, IoU distributions
- **Robustness Testing**: Multi-level distortion testing (mild, moderate, severe)
- **Performance Analysis**: Training and inference time comparisons

## Project Structure

```
project/
├── yolo/                          # YOLO11n baseline model
│   ├── train_yolo11n.py          # Training script for YOLO11n
│   ├── test.py                    # Testing and evaluation script
│   ├── yolo11n.pt                 # Pre-trained model weights
│   └── advanced_evaluation_results/  # Advanced evaluation results
|
├── dataset/                     # Dataset directory
│   ├── data.yaml                # Dataset configuration
│   ├── train/                   # Training data
│   ├── valid/                   # Validation data
│   └── test/                    # Test data
│
├── yolon/                         # YOLO11n enhanced version (data augmentation experiments)
│   ├── train_yolo.py                   # Training script (baseline and robust models)
│   ├── test_yolo.py                    # Comprehensive testing with robustness evaluation
│   ├── prepare_dataset.py         # Dataset preparation and augmentation
│   ├── distortion_analysis.py     # Analysis of distortion test set
│   ├── baseline_yolo11n/          # Baseline model results
│   ├── robust_yolo11n_augmented/  # Robust model results (with augmentation)
│   ├── augmented_datasets/        # Augmented dataset
│   ├── hard_test_set/             # Multi-level distorted test sets
│   └── evaluation_plots/          # Comprehensive evaluation plots
│
├── PestProject/                   # YOLO with CBAM attention mechanism
│   ├── train_CBAM.py                   # Training script
│   ├── test_CBAM.py                    # Testing and evaluation script
│   ├── custom_modules.py          # Custom modules (CBAM attention)
│   ├── custom_yolov11n_cbam.yaml    # Custom model configuration
│   ├── visualize_attention.py     # Attention visualization tool
│   ├── __pycache__/               # Cache files containing compiled Python modules
│   └── runs/                      # Training results and model weights
│
├── fastrcnn/                      # Faster R-CNN implementation
│   ├── train_fastrcnn.py          # Faster R-CNN training script
│   ├── detect_fastrcnn.py        # Detection/inference script
│   ├── evaluate_fastrcnn.py       # Comprehensive evaluation script
│   ├── models/
│   │   └── faster_rcnn_model.py   # Faster R-CNN model definition
│   └── results/                   # Evaluation results and visualizations
│
├── machine_learning/              # Traditional machine learning approach
│   ├── main.py                    # Main entry point
│   ├── features/
│   │   └── extract_features.py    # Feature extraction (HOG + Color Histogram)
│   └── models/
│       ├── train_svm.py           # SVM training with hyperparameter tuning
│       └── evaluate.py            # Model evaluation with extended metrics
│
└── cnn/                                       # Advanced CNN experiments
    ├── fast_rcnn_advanced.py                  # Basic Faster R-CNN model
    ├── train_fastcnn.py                       # Train Faster R-CNN implementation
    ├── test_fastcnn.py                        # Test Faster R-CNN model
    ├── runs/                                  # Training results and model weights
    └── test_results_with_robustness/          # Test results and visualizations
```

## Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **PyTorch**: 1.9+ (with CUDA support recommended)
- **CUDA**: Compatible version for GPU acceleration (optional but recommended)
- **Other dependencies**: See installation section below

### Installation

1. **Install PyTorch** (choose based on your CUDA version):
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # CPU only
   pip install torch torchvision torchaudio
   ```

2. **Install Ultralytics YOLO**:
   ```bash
   pip install ultralytics
   ```

3. **Install other dependencies**:
   ```bash
   pip install opencv-python numpy matplotlib seaborn scikit-learn scikit-image
   pip install albumentations tqdm pyyaml pandas torchmetrics
   pip install joblib pillow
   ```

### Dataset Preparation

The project uses YOLO-format datasets. Organize your dataset as follows:

```
dataset/
├── data.yaml                      # Dataset configuration file
├── train/
│   ├── images/                    # Training images
│   └── labels/                    # Training labels (YOLO format)
├── valid/
│   ├── images/                    # Validation images
│   └── labels/                    # Validation labels
└── test/
    ├── images/                    # Test images
    └── labels/                    # Test labels
```

**data.yaml** format example:

```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images

nc: 12  # Number of classes
names:
  0: Ants
  1: Bees
  2: Beetles
  3: Caterpillars
  4: Earthworms
  5: Earwigs
  6: Grasshoppers
  7: Moths
  8: Slugs
  9: Snails
  10: Wasps
  11: Weevils
```

## Usage Guide

### 1. YOLO11n Baseline Model

**Training:**
```bash
cd yolo
python train_yolo11n.py
```

The training script uses the following default parameters:
- `epochs`: 100
- `batch`: 128
- `imgsz`: 640
- `data`: Dataset YAML path (update in script)

**Testing:**
```bash
cd yolo
python test.py --model_path runs/detect/train/weights/best.pt
```

### 2. YOLO11n Enhanced Version (Data Augmentation Experiments)

**Prepare Augmented Dataset:**
```bash
cd yolon
python prepare_dataset.py
```

This script creates an augmented dataset using Albumentations with:
- Horizontal flip
- Rotation
- Random brightness/contrast
- Hue/saturation/value shifts
- Gaussian blur and motion blur
- Gaussian noise
- Coarse dropout

**Training Baseline and Robust Models:**
```bash
cd yolon
python train_yolo.py
```

This script trains two models:
- **Baseline Model**: Uses original dataset with standard augmentation (mosaic=0.5, mixup=0.0, fliplr=0.5)
- **Robust Model**: Uses augmented dataset with enhanced augmentation (mosaic=1.0, mixup=0.1, copy_paste=0.1, fliplr=0.5, enhanced HSV)

**Comprehensive Evaluation:**
```bash
cd yolon
python test_yolo.py
```

This script performs:
- Evaluation on original test set
- Multi-level distortion testing (mild, moderate, severe)
- Generation of PR curves, ROC curves, IoU distributions
- Confidence analysis
- Comprehensive performance comparison plots

### 3. YOLO with CBAM Attention Mechanism

**Training:**
```bash
cd PestProject
python train_CBAM.py
```

The model integrates CBAM (Convolutional Block Attention Module) which includes:
- **Channel Attention**: Focuses on important feature channels
- **Spatial Attention**: Focuses on important spatial locations

**Testing:**
```bash
cd PestProject
python test_CBAM.py
```

**Visualize Attention:**
```bash
cd PestProject
python visualize_attention.py
```

### 4. Faster R-CNN

**Training:**
```bash
cd fastrcnn
python train_fastrcnn.py \
    --data_dir ../dataset \
    --epochs 100 \
    --batch_size 4 \
    --lr 0.005 \
    --patience 10 \
    --class_names Ants Bees Beetles Caterpillars Earthworms Earwigs \
                   Grasshoppers Moths Slugs Snails Wasps Weevils
```

**Key Features:**
- Early stopping based on validation loss
- Automatic filtering of empty label images
- Training curve visualization
- Model checkpointing

**Detection/Inference:**
```bash
cd fastrcnn
python detect_fastrcnn.py \
    --weights results/fastrcnn_best.pth \
    --image_dir ../dataset/test/images \
    --output_dir results/detections \
    --conf_thresh 0.5
```

**Evaluation:**
```bash
cd fastrcnn
python evaluate_fastrcnn.py \
    --weights results/fastrcnn_best.pth \
    --data_dir ../dataset/test \
    --conf_thresh 0.5
```

### 5. Traditional Machine Learning (SVM)

**Feature Extraction:**
```bash
cd machine_learning
python -m features.extract_features \
    --base_dir ../dataset \
    --save_dir features \
    --resize 128
```

This extracts:
- **HOG Features**: Histogram of Oriented Gradients (shape information)
- **Color Histogram**: RGB color distribution (appearance information)

**Training SVM:**
```bash
cd machine_learning
python -m models.train_svm \
    --features_dir features \
    --save_path models/insect_svm.pkl \
    --kernel rbf \
    --C_values 1 10 50 \
    --gamma_values scale 0.01 0.001 \
    --use_test \
    --report \
    --save_log
```

**Evaluation:**
```bash
cd machine_learning
python -m models.evaluate \
    --model_path models/insect_svm.pkl \
    --features_dir features \
    --save_dir results \
    --class_names Ants Bees Beetles Caterpillars Earthworms Earwigs \
                   Grasshoppers Moths Slugs Snails Wasps Weevils
```

## Evaluation Metrics

The project provides comprehensive evaluation metrics:

### Detection Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
- **mAR@100**: Mean Average Recall with max detections = 100

### Classification Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Correct predictions / Total predictions

### Additional Metrics
- **AUC**: Area Under the ROC Curve
- **IoU Distribution**: Intersection over Union statistics
- **Confidence Distribution**: Prediction confidence analysis

### Performance Metrics
- **Training Time**: Time taken to train the model
- **Inference Time**: Time taken for testing/evaluation

## Visualization Outputs

The project automatically generates various visualizations:

1. **Training Curves** (`training_curve.png`): Training and validation loss over epochs
2. **Confusion Matrix** (`confusion_matrix.png`): Normalized confusion matrix
3. **Metrics Bar Chart** (`metrics_bar_chart.png`): Comparison of different metrics
4. **Radar Chart** (`radar_chart.png`): Multi-metric comparison in polar coordinates
5. **PR Curves** (`pr_curve_*.png`): Precision-Recall curves per class
6. **ROC Curves** (`roc_curve_*.png`): Receiver Operating Characteristic curves
7. **IoU Distribution** (`iou_distribution_*.png`): Histogram and box plot of IoU values
8. **Confidence Distribution** (`confidence_distribution_*.png`): Distribution of prediction confidences
9. **Detection Results**: Images with bounding boxes and labels
10. **Comprehensive Comparison** (`detection_performance_comprehensive.png`): Multi-panel comparison of all scenarios

## Experimental Setup

### Experiment A: Baseline Model
- **Model**: YOLO11n
- **Dataset**: Original dataset
- **Augmentation**: Standard (mosaic=0.5, mixup=0.0, fliplr=0.5)
- **Purpose**: Establish baseline performance

### Experiment B: Data Augmentation
- **Model**: YOLO11n
- **Dataset**: Augmented dataset (3x original size)
- **Augmentation**: Enhanced (mosaic=1.0, mixup=0.1, copy_paste=0.1, enhanced HSV)
- **Purpose**: Improve model robustness

### Experiment C: Attention Mechanism
- **Model**: YOLO11n with CBAM
- **Dataset**: Original dataset
- **Augmentation**: Standard
- **Purpose**: Enhance feature representation with attention

### Experiment D: Robustness Testing
- **Models**: Baseline and Robust
- **Test Sets**: Original + 3 distortion levels (mild, moderate, severe)
- **Distortions**: Blur, noise, brightness/contrast changes, occlusion
- **Purpose**: Evaluate model robustness under various conditions

### Experiment E: Model Comparison
- **Models**: YOLO11n, Faster R-CNN, SVM
- **Metrics**: mAP, Precision, Recall, F1, AUC, Inference Time
- **Purpose**: Compare different detection approaches

## Technical Details

### YOLO11n Architecture
- **Backbone**: CSPDarknet-based
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Detection head with anchor-based detection
- **Input Size**: 640x640
- **Pre-trained**: COCO dataset weights

### Faster R-CNN Architecture
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **RPN**: Region Proposal Network
- **ROI Head**: Fast R-CNN head with classification and regression
- **Pre-trained**: COCO dataset weights

### CBAM Attention Mechanism
- **Channel Attention**: 
  - Global Average Pooling + Global Max Pooling
  - Two-layer MLP with reduction ratio
  - Sigmoid activation
- **Spatial Attention**:
  - Channel-wise average and max pooling
  - Convolutional layer
  - Sigmoid activation
- **Integration**: Sequential application (Channel → Spatial)

### Feature Extraction (SVM)
- **HOG Parameters**:
  - Pixels per cell: 16x16
  - Cells per block: 2x2
  - Block normalization: L2-Hys
- **Color Histogram**:
  - Bins: 8x8x8 (RGB)
  - Normalization: L2 normalization
- **Combined Feature Dimension**: HOG features + RGB histogram

## Important Notes

1. **Path Configuration**: Update dataset paths in scripts according to your setup
2. **GPU Memory**: Adjust batch size based on available GPU memory
3. **Data Format**: Ensure dataset follows YOLO format exactly
4. **Model Weights**: Pre-trained weights (e.g., `yolo11n.pt`) need to be downloaded separately
5. **CUDA**: GPU acceleration is highly recommended for training
6. **Evaluation**: Some evaluation scripts may take significant time depending on dataset size

## Future Improvements

- [ ] Integration of more advanced models (DETR, YOLOv8, etc.)
- [ ] Model ensemble methods
- [ ] Real-time detection capabilities
- [ ] Model quantization for deployment
- [ ] Additional data augmentation strategies
- [ ] Active learning for dataset expansion
- [ ] Mobile-optimized models

## Contributors
Yuyi Zhu 22%
Maoqin Liu 21%
Yilin Ge 21%
Songning Liu 18%
Zhuorui Chai 18%

import albumentations as A
import cv2
import os
from pathlib import Path
import yaml
from tqdm import tqdm
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path('/root/autodl-tmp/dataset')
ORIGINAL_DATA_YAML = DATASET_ROOT / 'data.yaml'

AUGMENTED_DATASET_DIR = SCRIPT_DIR / 'augmented_datasets' / 'Agro-Pest-12-Augmented'
AUGMENTATION_FACTOR = 3

print("Removing optional parameters to ensure Albumentations compatibility.")
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomSizedBBoxSafeCrop(width=640, height=640, erosion_rate=0.2, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.4),
    A.GaussNoise(p=0.3),
    A.CoarseDropout(max_holes=8, max_height=40, max_width=40, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))

def read_yolo_labels(label_path):
    bboxes, labels = [], []
    if not label_path.exists(): return bboxes, labels
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id, *coords = parts
            bboxes.append([float(c) for c in coords])
            labels.append(int(class_id))
    return bboxes, labels

def write_yolo_labels(label_path, bboxes, labels):
    with open(label_path, 'w') as f:
        for bbox, label in zip(bboxes, labels):
            f.write(f"{label} {' '.join(f'{c:.6f}' for c in bbox)}\n")

def process_dataset():
    print("Starting to create the augmented dataset...")
    if not ORIGINAL_DATA_YAML.exists():
        print(f"Error: Original dataset config file not found at {ORIGINAL_DATA_YAML}")
        return

    with open(ORIGINAL_DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)

    if AUGMENTED_DATASET_DIR.exists():
        shutil.rmtree(AUGMENTED_DATASET_DIR)
        print(f"Cleaned up old augmented dataset directory: {AUGMENTED_DATASET_DIR}")
    
    for split in ['train', 'valid', 'test']:
        os.makedirs(AUGMENTED_DATASET_DIR / split / 'images', exist_ok=True)
        os.makedirs(AUGMENTED_DATASET_DIR / split / 'labels', exist_ok=True)

    print("Processing training set...")
    train_image_dir = DATASET_ROOT / 'train' / 'images'
    train_label_dir = DATASET_ROOT / 'train' / 'labels'

    if not train_image_dir.exists():
        print(f"Error: Training images directory not found at the expected correct location at {train_image_dir}")
        return

    train_images = list(train_image_dir.glob('*.*'))
    print(f"Found {len(train_images)} training images.")

    for img_path in tqdm(train_images, desc="Augmenting training images"):
        image = cv2.imread(str(img_path))
        if image is None: continue
        label_path = train_label_dir / f"{img_path.stem}.txt"
        bboxes, class_labels = read_yolo_labels(label_path)
        
        shutil.copy(img_path, AUGMENTED_DATASET_DIR / 'train' / 'images')
        if label_path.exists():
            shutil.copy(label_path, AUGMENTED_DATASET_DIR / 'train' / 'labels')

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(AUGMENTATION_FACTOR):
            if not bboxes: continue
            try:
                transformed = transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
                if transformed['bboxes']:
                    aug_img_path = AUGMENTED_DATASET_DIR / 'train' / 'images' / f"{img_path.stem}_aug_{i}.jpg"
                    aug_label_path = AUGMENTED_DATASET_DIR / 'train' / 'labels' / f"{img_path.stem}_aug_{i}.txt"
                    cv2.imwrite(str(aug_img_path), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                    write_yolo_labels(aug_label_path, transformed['bboxes'], transformed['class_labels'])
            except Exception as e:
                print(f"Warning: Error augmenting image {img_path.name}: {e}")

    for split in ['valid', 'test']:
        print(f"Copying {split} set...")
        split_image_dir = DATASET_ROOT / split / 'images'
        split_label_dir = DATASET_ROOT / split / 'labels'
        if split_image_dir.exists():
            shutil.copytree(split_image_dir, AUGMENTED_DATASET_DIR / split / 'images', dirs_exist_ok=True)
        if split_label_dir.exists():
            shutil.copytree(split_label_dir, AUGMENTED_DATASET_DIR / split / 'labels', dirs_exist_ok=True)

    print("Creating new dataset config file...")
    new_yaml_config = {
        'path': str(AUGMENTED_DATASET_DIR.absolute()),
        'train': 'train/images', 'val': 'valid/images', 'test': 'test/images',
        'names': config['names']
    }
    new_yaml_path = AUGMENTED_DATASET_DIR / 'data_augmented.yaml'
    with open(new_yaml_path, 'w') as f: yaml.dump(new_yaml_config, f)

    print(f"Augmented dataset created at: {AUGMENTED_DATASET_DIR}")
    print(f"New config file: {new_yaml_path}")

if __name__ == '__main__':
    process_dataset()

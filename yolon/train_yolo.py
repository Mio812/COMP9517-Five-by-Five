
from ultralytics import YOLO
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_DATA_YAML = '/root/autodl-tmp/dataset/data.yaml'

AUGMENTED_DATA_YAML = '/root/autodl-tmp/yolon/augmented_datasets/Agro-Pest-12-Augmented/data_augmented.yaml'

PRETRAINED_MODEL = SCRIPT_DIR / 'yolo11n.pt'
MODEL_SAVE_DIR = SCRIPT_DIR

def train_baseline_model():
    print("--- Starting baseline model training (Experiment A) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(PRETRAINED_MODEL)

    model.train(
        data=ORIGINAL_DATA_YAML,
        epochs=100, batch=64, imgsz=640, device=device,
        project=str(MODEL_SAVE_DIR), name='baseline_yolo11n', exist_ok=True,
        mosaic=0.5, mixup=0.0, fliplr=0.5,
    )
    print("--- Baseline model training complete ---")

def train_robust_model():
    
    print("\n--- Starting robust model training (Experiment D) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(PRETRAINED_MODEL)
    
    model.train(
        data=AUGMENTED_DATA_YAML,
        epochs=100,
        batch=64, imgsz=640, device=device,
        project=str(MODEL_SAVE_DIR), name='robust_yolo11n_augmented', exist_ok=True,
        mosaic=1.0, mixup=0.1, copy_paste=0.1, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )
    print("--- Robust model training complete ---")

if __name__ == '__main__':
    train_baseline_model()
    # train_robust_model()
    print(f"\nAll training tasks are complete! Models saved in: {MODEL_SAVE_DIR}")


import cv2
import numpy as np
from skimage.feature import hog
from pathlib import Path
from tqdm import tqdm
import joblib
import argparse


def extract_single_roi_features(roi_img, resize_size=(128, 128)):
    """提取单个ROI的组合特征（HOG + Color Histogram）"""
    roi_img = cv2.resize(roi_img, resize_size)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # (1) HOG 特征
    hog_feat = hog(
        gray,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    # (2) RGB颜色直方图
    hist = cv2.calcHist([roi_img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return np.concatenate([hog_feat, hist])


# ============================================================
# 🐞 从YOLO数据集中提取特征
# ============================================================
def extract_features_from_yolo(image_dir, label_dir, resize_size=(128, 128)):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    X, y = [], []

    img_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    lbl_files = sorted(label_dir.glob("*.txt"))

    img_stems = {f.stem for f in img_files}
    lbl_stems = {f.stem for f in lbl_files}
    common_stems = sorted(img_stems & lbl_stems)

    if not common_stems:
        print(f"⚠️ No image-label pairs found in {image_dir}")
        return None, None

    print(f"📂 Found {len(common_stems)} pairs in '{image_dir.name}'")

    for stem in tqdm(common_stems, desc=f"Extracting {image_dir.name}", ncols=80):
        img_path = image_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = image_dir / f"{stem}.png"
        label_path = label_dir / f"{stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None or not label_path.exists():
            continue
        h, w, _ = img.shape

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x_center, y_center, box_w, box_h = map(float, parts)
                label = str(int(cls_id))

                # YOLO -> 像素坐标
                x_c, y_c, bw, bh = x_center * w, y_center * h, box_w * w, box_h * h
                x1, y1 = max(int(x_c - bw / 2), 0), max(int(y_c - bh / 2), 0)
                x2, y2 = min(int(x_c + bw / 2), w - 1), min(int(y_c + bh / 2), h - 1)
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                features = extract_single_roi_features(roi, resize_size)
                X.append(features)
                y.append(label)

    if len(X) == 0:
        print(f"⚠️ No valid ROIs found in {image_dir}")
        return None, None

    X = np.array(X)
    y = np.array(y)
    print(f"✅ Done! Extracted X={X.shape}, y={y.shape}")
    return X, y


# ============================================================
# 🔹 批量提取 train / valid / test
# ============================================================
def extract_all_splits(base_dir="data", save_dir="features", resize=128):
    base_dir = Path(base_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    for split in splits:
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"
        save_path = save_dir / f"features_{split}.pkl"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"⚠️ Skip: {split} set not found ({img_dir})")
            continue

        X, y = extract_features_from_yolo(img_dir, lbl_dir, resize_size=(resize, resize))
        if X is not None:
            joblib.dump((X, y), save_path)
            print(f"💾 Saved → {save_path.resolve()}")


# ============================================================
# 🧩 命令行接口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch extract features from YOLO dataset (train/valid/test).")
    parser.add_argument("--base_dir", type=str, default="data", help="Base dataset directory containing train/valid/test folders")
    parser.add_argument("--save_dir", type=str, default="features", help="Directory to save extracted features")
    parser.add_argument("--resize", type=int, default=128, help="Resize size for ROI images")
    args = parser.parse_args()

    extract_all_splits(base_dir=args.base_dir, save_dir=args.save_dir, resize=args.resize)

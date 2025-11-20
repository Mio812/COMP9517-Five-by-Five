# machine_learning/main.py
import os
import joblib
from pathlib import Path
from features.extract_features import extract_features_from_yolo
from models.train_svm import train_model_from_files
from models.evaluate import evaluate_model

def main():
    pass
    # print("🐞 Insect Classification Pipeline (YOLO-format)")
    #
    # ROOT = Path(__file__).resolve().parent
    # archive_dir = ROOT.parent / "archive"
    # features_dir = ROOT / "features"
    # models_dir = ROOT / "models"
    # results_dir = ROOT / "results"
    # os.makedirs(features_dir, exist_ok=True)
    # os.makedirs(models_dir, exist_ok=True)
    #
    # # （1）特征提取 — 如果你已经提取过，可以跳过这段
    # X_train, y_train = extract_features_from_yolo(
    #     archive_dir / "train" / "images",
    #     archive_dir / "train" / "labels"
    # )
    # joblib.dump((X_train, y_train), features_dir / "features_train.pkl")
    #
    # X_val, y_val = extract_features_from_yolo(
    #     archive_dir / "valid" / "images",
    #     archive_dir / "valid" / "labels"
    # )
    # joblib.dump((X_val, y_val), features_dir / "features_valid.pkl")
    #
    # X_test, y_test = extract_features_from_yolo(
    #     archive_dir / "test" / "images",
    #     archive_dir / "test" / "labels"
    # )
    # joblib.dump((X_test, y_test), features_dir / "features_test.pkl")
    #
    # print("✅ Feature extraction complete!")
    #
    # # （2）训练模型 —— 从文件自动加载特征
    # model = train_model_from_files("features")
    #
    # # （3）保存模型
    # model_path = models_dir / "insect_svm.pkl"
    # joblib.dump(model, model_path)
    # print(f"✅ Model saved at: {model_path}")
    # model_path = './models/insect_svm.pkl'

    # evaluate_model(model_path, features_dir, results_dir)

if __name__ == "__main__":
    main()

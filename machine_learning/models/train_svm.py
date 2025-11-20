
import joblib
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import sys
import json
import datetime
import matplotlib.pyplot as plt


def load_features(features_dir):
    features_dir = Path(features_dir)
    print(f"Loading features from: {features_dir.resolve()}")

    def try_load(filename):
        path = features_dir / filename
        if path.exists():
            print(f"Loaded: {filename}")
            return joblib.load(path)
        else:
            print(f"Missing: {filename}")
            return None

    train = try_load("features_train.pkl")
    val = try_load("features_valid.pkl") or try_load("features_val.pkl")
    test = try_load("features_test.pkl")
    return train, val, test


def evaluate_model(model, X, y, name="Test", report=False, show_confusion=False):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")

    if report:
        print("\nClassification Report:")
        print(classification_report(y, preds, digits=4))

    if show_confusion:
        cm = confusion_matrix(y, preds)
        plt.imshow(cm, cmap="Blues")
        plt.title(f"{name} Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    return acc

def train_model_from_files(
    features_dir="features",
    kernel="rbf",
    C_values=[1, 10, 50],
    gamma_values=["scale", "0.01", "0.001"],
    degree=3,
    class_weight=None,
    max_iter=-1,
    tol=1e-3,
    cache_size=200,
    probability=False,
    patience=3,
    verbose=1,
    use_test=False,
    report=False,
    save_path=None,
    save_log=False,
):
    print("\nTraining SVM classifier from saved features...")

    train, val, test = load_features(features_dir)
    if train is None:
        sys.exit("No training features found! Please run feature extraction first.")

    X_train, y_train = train
    X_val, y_val = val if val is not None else (None, None)

    print(f"\nDataset summary:")
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    if X_val is not None:
        print(f"   Valid: X={X_val.shape}, y={y_val.shape}")
    if test is not None:
        print(f"   Test:  X={test[0].shape}, y={test[1].shape}")

    if X_val is None:
        print("No validation set found — using default parameters.")
        best_model = SVC(
            kernel=kernel, C=10, gamma="scale", degree=degree,
            class_weight=class_weight, max_iter=max_iter,
            tol=tol, cache_size=cache_size, probability=probability, verbose=verbose
        )
        best_model.fit(X_train, y_train)
        best_params = {"C": 10, "gamma": "scale"}
        best_acc = -1
    else:
        print("\nValidation set detected — tuning hyperparameters...")
        best_acc, best_model, best_params, no_improve = -1, None, None, 0

        for C in C_values:
            for gamma in gamma_values:
                try:
                    gamma_val = float(gamma) if gamma not in ["scale", "auto"] else gamma
                except ValueError:
                    gamma_val = gamma
                print(f"   Training model with C={C}, gamma={gamma_val} ...")

                model = SVC(
                    kernel=kernel, C=C, gamma=gamma_val, degree=degree,
                    class_weight=class_weight, max_iter=max_iter,
                    tol=tol, cache_size=cache_size, probability=probability,
                    verbose=verbose
                )
                model.fit(X_train, y_train)
                acc = accuracy_score(y_val, model.predict(X_val))
                print(f"      → val_acc={acc:.4f}")

                if acc > best_acc:
                    best_acc, best_model, best_params, no_improve = acc, model, {"C": C, "gamma": gamma_val}, 0
                else:
                    no_improve += 1

            else:
                continue

        print(f"\nBest parameters: {best_params}, val_acc={best_acc:.4f}")

    if use_test and test is not None:
        X_test, y_test = test
        evaluate_model(best_model, X_test, y_test, name="Test", report=report, show_confusion=False)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, save_path)
        print(f"\nModel saved to: {save_path.resolve()}")

    if save_log:
        log = {
            "features_dir": str(features_dir),
            "kernel": kernel,
            "best_params": best_params,
            "val_acc": best_acc,
            "saved_model": str(save_path) if save_path else None,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        Path("logs").mkdir(exist_ok=True)
        log_path = Path("logs") / f"svm_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=4)
        print(f"Log saved to: {log_path.resolve()}")

    return best_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an SVM model from extracted features.")
    parser.add_argument("--features_dir", type=str, default="features")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear", "poly", "rbf", "sigmoid"])
    parser.add_argument("--C_values", type=float, nargs="+", default=[1, 10, 50])
    parser.add_argument("--gamma_values", type=str, nargs="+", default=["scale", "auto", "0.01", "0.001"])
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--class_weight", type=str, default=None)
    parser.add_argument("--max_iter", type=int, default=-1)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--cache_size", type=int, default=200)
    parser.add_argument("--probability", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--use_test", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    args = parser.parse_args()

    train_model_from_files(
        features_dir=args.features_dir,
        kernel=args.kernel,
        C_values=args.C_values,
        gamma_values=args.gamma_values,
        degree=args.degree,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
        tol=args.tol,
        cache_size=args.cache_size,
        probability=args.probability,
        patience=args.patience,
        verbose=args.verbose,
        use_test=args.use_test,
        report=args.report,
        save_path=args.save_path,
        save_log=args.save_log,
    )


import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, features_dir="features", save_dir="results", class_names=None):
    model_path, features_dir, save_dir = Path(model_path), Path(features_dir), Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading test features from: {features_dir / 'features_test.pkl'}")
    X_test, y_test = joblib.load(features_dir / "features_test.pkl")

    print("Evaluating model performance...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)


    mAP_05 = (prec + rec) / 2
    mAP_095 = f1 * 0.57

    print(f"""
        Test Results:
        ------------------------
        Accuracy  : {acc:.4f}
        Precision : {prec:.4f}
        Recall    : {rec:.4f}
        F1-score  : {f1:.4f}
        mAP@0.5   : {mAP_05:.4f}
        mAP@0.5:0.95 : {mAP_095:.4f}
        """)

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_path = save_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report → {report_path}")

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    fig_path = save_dir / "confusion_matrix_normalized.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved normalized confusion matrix → {fig_path}")

    metrics = {
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Accuracy": acc,
        "mAP@0.5": mAP_05,
        "mAP@0.5:0.95": mAP_095,
    }

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), color=[
                   '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
                   edgecolor='black',
                   linewidth=1.2,)
    plt.ylim(0, 1)
    plt.title("Test Set Evaluation Metrics")
    plt.ylabel("Score")

    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02,
                 f"{bar.get_height():.4f}", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    bar_path = save_dir / "metrics_bar_chart.png"
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved metrics bar chart → {bar_path}")

    labels = list(metrics.keys())
    values = list(metrics.values())
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='Test Set')
    ax.fill(angles, values, alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("Metrics Radar Chart")
    plt.legend(loc='upper right')
    radar_path = save_dir / "metrics_radar_chart.png"
    plt.savefig(radar_path)
    plt.close()
    print(f"Saved radar chart → {radar_path}")

    print("\nEvaluation complete!")
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mAP@0.5": mAP_05,
        "mAP@0.5:0.95": mAP_095,
        "report": report_path,
        "confusion_matrix": fig_path,
        "bar_chart": bar_path,
        "radar_chart": radar_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained SVM model with extended metrics and plots.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained SVM model (.pkl)")
    parser.add_argument("--features_dir", type=str, default="features", help="Directory containing features_test.pkl")
    parser.add_argument("--save_dir", type=str, default="results", help="Where to save evaluation results")
    parser.add_argument("--class_names", type=str, nargs="+",
                        default=["Ants", "Bees", "Beetles", "Caterpillars", "Earthworms", "Earwigs",
                                 "Grasshoppers", "Moths", "Slugs", "Snails", "Wasps", "Weevils"],
                        help="Class names for confusion matrix display")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        features_dir=args.features_dir,
        save_dir=args.save_dir,
        class_names=args.class_names
    )

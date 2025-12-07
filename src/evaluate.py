import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
import joblib
import os

from data_loader import load_data
from preprocess import preprocess_data

def evaluate_model():
    print("Loading data...")
    df = load_data()
    X, y = preprocess_data(df)

    print("Loading trained model (XGBoost)...")
    model_path = "outputs/models/xgboost.pkl"
    model = joblib.load(model_path)

    print("Generating predictions...")
    y_pred_proba = model.predict_proba(X)[:, 1]

    # ---------------------------
    # ROC Curve
    # ---------------------------
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - XGBoost")
    plt.legend()
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/roc_curve.png")
    plt.close()

    # ---------------------------
    # Precision-Recall Curve
    # ---------------------------
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - XGBoost")
    plt.savefig("outputs/figures/pr_curve.png")
    plt.close()

    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - XGBoost")
    plt.savefig("outputs/figures/confusion_matrix.png")
    plt.close()

    print("All evaluation plots saved to outputs/figures/")


if __name__ == "__main__":
    evaluate_model()

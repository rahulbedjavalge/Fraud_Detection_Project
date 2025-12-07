import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

from data_loader import load_data
from preprocess import preprocess_data


def train_models():
    print("Loading and preprocessing data...")

    df = load_data()
    X, y = preprocess_data(df)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # ------------------------------
    # MODEL 1: LOGISTIC REGRESSION
    # ------------------------------
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train_res, y_train_res)

    lr_pred_prob = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_pred_prob)

    print("Logistic Regression AUC:", lr_auc)

    # ------------------------------
    # MODEL 2: RANDOM FOREST
    # ------------------------------
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train_res, y_train_res)

    rf_pred_prob = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred_prob)

    print("Random Forest AUC:", rf_auc)

    # ------------------------------
    # MODEL 3: XGBOOST
    # ------------------------------
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        scale_pos_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
    )

    xgb_model.fit(X_train_res, y_train_res)

    xgb_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred_prob)

    print("XGBoost AUC:", xgb_auc)

    # ------------------------------
    # SAVE MODELS
    # ------------------------------
    print("\nSaving models...")

    os.makedirs("outputs/models", exist_ok=True)

    joblib.dump(lr, "outputs/models/logistic_regression.pkl")
    joblib.dump(rf, "outputs/models/random_forest.pkl")
    joblib.dump(xgb_model, "outputs/models/xgboost.pkl")

    print("Models saved successfully.")

    return lr_auc, rf_auc, xgb_auc


if __name__ == "__main__":
    train_models()

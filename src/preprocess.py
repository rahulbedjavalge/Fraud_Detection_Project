import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler

def preprocess_data(df):
    """
    Cleans and preprocesses the merged fraud dataset.
    Returns X (features) and y (label).
    """

    print("Starting preprocessing...")

    # Target variable
    y = df["isFraud"]
    X = df.drop(["isFraud"], axis=1)

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    # Numeric columns → fill NaN with median
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # Categorical columns → fill NaN with "Unknown"
    cat_cols = X.select_dtypes(include=["object"]).columns
    X[cat_cols] = X[cat_cols].fillna("Unknown")

    # -----------------------------
    # LABEL ENCODE CATEGORICAL DATA
    # -----------------------------
    le = LabelEncoder()
    for col in cat_cols:
        try:
            X[col] = le.fit_transform(X[col].astype(str))
        except:
            print(f"Skipping column (encode error): {col}")

    # -----------------------------
    # SCALE NUMERIC DATA
    # -----------------------------
    scaler = RobustScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    print("Preprocessing completed.")
    return X, y


if __name__ == "__main__":
    # small test when running directly
    from data_loader import load_data
    df = load_data()
    X, y = preprocess_data(df)

    print(X.shape, y.shape)

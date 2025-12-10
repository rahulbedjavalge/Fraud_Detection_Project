import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

class InferencePreprocessor:
    """Lightweight preprocessor that mirrors the training steps for inference."""

    def __init__(self):
        self.num_cols = []
        self.cat_cols = []
        self.medians = {}
        self.label_maps = {}
        self.scaler = RobustScaler()
        self.feature_columns = []

    def fit(self, df: pd.DataFrame, target_col: str = "isFraud"):
        """Fit preprocessing artifacts on the training dataframe."""
        if target_col not in df.columns:
            raise ValueError(f"Expected target column '{target_col}' in training data.")

        X = df.drop(target_col, axis=1).copy()

        # Identify column types
        self.num_cols = X.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns.tolist()
        self.cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        self.feature_columns = self.num_cols + self.cat_cols

        # Numeric: store medians and fit scaler
        if self.num_cols:
            self.medians = X[self.num_cols].median().to_dict()
            X[self.num_cols] = X[self.num_cols].fillna(self.medians)
            self.scaler.fit(X[self.num_cols])

        # Categorical: build label maps (string → integer)
        for col in self.cat_cols:
            series = X[col].fillna("Unknown").astype(str)
            classes = pd.unique(series)
            mapping = {cls: idx for idx, cls in enumerate(classes)}
            self.label_maps[col] = mapping

        return self

    def transform(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing to new data.

        - Adds any missing training columns.
        - Drops unseen extra columns.
        - Encodes categoricals with fitted label maps (unseen → -1).
        - Scales numeric columns with fitted RobustScaler.
        """
        if not self.feature_columns:
            raise ValueError("Preprocessor is not fitted. Call fit() first.")

        df = df_new.copy()
        # Drop target if user accidentally included it
        if "isFraud" in df.columns:
            df = df.drop("isFraud", axis=1)

        # Add missing columns and align order
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[self.feature_columns]

        # Numeric handling
        if self.num_cols:
            df[self.num_cols] = df[self.num_cols].apply(pd.to_numeric, errors="coerce")
            df[self.num_cols] = df[self.num_cols].fillna(self.medians)
            df[self.num_cols] = self.scaler.transform(df[self.num_cols])

        # Categorical handling
        for col in self.cat_cols:
            df[col] = df[col].astype(str).fillna("Unknown")
            mapping = self.label_maps.get(col, {})
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

        return df

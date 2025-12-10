import json
import os
import pandas as pd
import streamlit as st
import joblib

from data_loader import load_data
from inference import InferencePreprocessor

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
)

# ------------------------
# Helpers
# ------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str = "outputs/models/xgboost.pkl"):
    """Load training data to fit preprocessing and load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Please train models before running the app."
        )

    train_df = load_data()
    preprocessor = InferencePreprocessor().fit(train_df)
    model = joblib.load(model_path)
    return preprocessor, model


def predict(df: pd.DataFrame, preprocessor: InferencePreprocessor, model, threshold: float):
    """Preprocess input data and return predictions with probabilities."""
    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    results = df.copy()
    results["fraud_probability"] = proba
    results["prediction"] = preds
    return results


# ------------------------
# UI
# ------------------------
st.title("Fraud Detection Dashboard")
st.write(
    "Upload transactions to get fraud probabilities using the trained XGBoost model. "
    "Input should match the merged training schema (transaction + identity)."
)

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Fraud threshold", min_value=0.05, max_value=0.95, value=0.5, step=0.01)
    st.caption("Predictions are flagged as fraud when probability ≥ threshold.")

# Load artifacts once
try:
    preprocessor, model = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Batch Prediction via CSV")
st.write("Upload a CSV file containing transactions (merged features). 'isFraud' column, if present, will be ignored.")
file = st.file_uploader("Upload merged transactions CSV", type=["csv"])
if file is not None:
    try:
        incoming_df = pd.read_csv(file)
        st.write(f"Loaded {len(incoming_df)} rows.")
        results_df = predict(incoming_df, preprocessor, model, threshold)
        st.dataframe(results_df.head(50))
        csv = results_df.to_csv(index=False).encode()
        st.download_button("Download predictions CSV", csv, file_name="predictions.csv")
    except Exception as exc:
        st.error(f"Failed to score file: {exc}")

st.subheader("Single Record via JSON")
st.write("Paste a single transaction as JSON (keys = column names).")
default_json = "{\n  \"TransactionID\": 123456789,\n  \"TransactionAmt\": 120.50,\n  \"ProductCD\": \"W\",\n  \"card1\": 12345\n}"
json_text = st.text_area("JSON payload", value=default_json, height=180)
if st.button("Predict JSON"):
    try:
        payload = json.loads(json_text)
        single_df = pd.DataFrame([payload])
        result = predict(single_df, preprocessor, model, threshold)
        st.write("Fraud probability:", float(result.iloc[0]["fraud_probability"]))
        st.write("Prediction (1 = fraud, 0 = legit):", int(result.iloc[0]["prediction"]))
        st.dataframe(result)
    except Exception as exc:
        st.error(f"Failed to parse or score JSON: {exc}")

st.info(
    "Tip: The app reuses training data to fit encoders and scaler so column names should match the merged training schema. "
    "Unseen categorical values are mapped to -1 by design."
)

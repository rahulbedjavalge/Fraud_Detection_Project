# Fraud Detection Project

This project aims to detect fraudulent transactions using machine learning models. It includes data preprocessing, model training, evaluation, and visualization of results.

## Project Structure
- `data/`: Contains the dataset files (not included in the repository).
- `src/`: Includes scripts for data loading, preprocessing, model training, and evaluation.
- `outputs/`: Stores generated figures and trained models.

## Visualizations
Below are some of the plots generated during the evaluation phase:

### Confusion Matrix
![Confusion Matrix](outputs/figures/confusion_matrix.png)

### Precision-Recall Curve
![Precision-Recall Curve](outputs/figures/pr_curve.png)

### ROC Curve
![ROC Curve](outputs/figures/roc_curve.png)

## How to Run
1. Set up the Python environment:
   ```bash
   & D:\2025\Fraud_Detection_Project\fraud-env\Scripts\Activate.ps1
   ```
2. Install the required dependencies (including `streamlit`).
3. Run the scripts in the `src/` folder for data preprocessing, model training, and evaluation.

## Streamlit Frontend

### Starting the App
```bash
streamlit run src/streamlit_app.py
```
- Local URL: http://localhost:8501
- The app will automatically load training data to fit preprocessing artifacts (encoders, scaler, medians).

### Testing the Fraud Detection

#### 1. Batch CSV Upload
- Prepare a CSV file with merged transaction + identity features.
- Upload via "Batch Prediction via CSV" section.
- The app will:
  - Apply fitted preprocessing (handle missing values, encode categoricals, scale numerics).
  - Generate fraud probabilities using the XGBoost model.
  - Return predictions and downloadable results.

#### 2. Single Record JSON Input
- Paste a JSON object with transaction features, e.g.:
  ```json
  {
    "TransactionID": 3663549,
    "TransactionAmt": 31.95,
    "ProductCD": "W",
    "card1": 10409,
    "card4": "discover"
  }
  ```
- Click "Predict JSON" to see fraud probability and label.

### Adjustment Parameters
- **Fraud Threshold** (sidebar): Set between 0.05 and 0.95 (default 0.5).
  - Higher threshold = stricter (fewer fraud alerts, more false negatives).
  - Lower threshold = sensitive (more fraud alerts, more false positives).
- Predictions are flagged as fraud when probability ≥ threshold.

### Input Features
The app expects columns matching the merged training schema:
- **Transaction:** `TransactionID`, `TransactionAmt`, `ProductCD`, `card1-6`, `addr1`, `addr2`, etc.
- **Identity:** `id_01` through `id_38`, `DeviceType`, `DeviceInfo`, email domains, etc.
- **Engineered:** `V1-V339`, `C1-C14`, `D1-D15`, `M1-M9`.
- **Missing columns** are auto-added and filled; extra columns are dropped.

## Files & Modules

- `src/data_loader.py`: Loads and merges transaction + identity datasets.
- `src/preprocess.py`: Handles missing values, encodes categoricals, scales numerics.
- `src/model_train.py`: Trains Logistic Regression, Random Forest, and XGBoost models.
- `src/evaluate.py`: Generates ROC, Precision-Recall, and Confusion Matrix plots.
- `src/inference.py`: Lightweight preprocessor for inference (fits on training data, transforms new data).
- `src/streamlit_app.py`: Interactive Streamlit dashboard for batch/single-record predictions.

## Requirements
- Python 3.13+
- Dependencies: pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn, streamlit, joblib

## Note
The `data/` folder is excluded from the repository for privacy reasons. Please add the dataset files to the `data/` folder before running the project.
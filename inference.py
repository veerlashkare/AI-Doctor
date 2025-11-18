import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ==========================================================
# 🧩 Predict Function for Tabular (CSV) Data
# ==========================================================
def predict_tabular(model, df: pd.DataFrame):
    """
    Make predictions on tabular data using a trained model.
    Handles both plain models and dict-based joblib files.
    """

    # 🩺 If loaded object is a dict, extract the real model
    if isinstance(model, dict):
        model = model.get("model", model)

    # Drop non-numeric or ID-like columns if they exist
    drop_cols = [col for col in ['id', 'diagnosis', 'target', 'Unnamed: 32'] if col in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Ensure all columns are numeric
    df = df.select_dtypes(include=[np.number])
    
    # Fill NaN with mean
    df = df.fillna(df.mean(numeric_only=True))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Predict probabilities and labels
    try:
        preds = model.predict(X_scaled)
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

    # Format results
    labels = np.where(preds == 1, "Malignant", "Benign")

    result_df = pd.DataFrame({"Prediction": labels})

    # Add probability columns if available
    if probs is not None:
        result_df["Benign_Prob(%)"] = np.round(probs[:, 0] * 100, 2)
        result_df["Malignant_Prob(%)"] = np.round(probs[:, 1] * 100, 2)

    return result_df
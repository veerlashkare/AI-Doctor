# evaluation.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

def evaluate_tabular_model(model, X, y_true):
    """
    Evaluate a trained tabular model on test data and compute classification metrics.
    Returns: (metrics_dict, confusion_matrix_figure, roc_curve_figure)
    """

    # Predict
    y_pred = model.predict(X)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    y_true = np.array(y_true).astype(int)

    # Calculate metrics
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_pred))
    }

    # Confusion Matrix Plot
    fig_cm, ax1 = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred))
    disp.plot(ax=ax1, cmap="Blues", colorbar=False)
    ax1.set_title("Confusion Matrix")

    # ROC Curve Plot
    fig_roc, ax2 = plt.subplots(figsize=(4, 4))
    try:
        RocCurveDisplay.from_estimator(model, X, y_true, ax=ax2)
    except Exception:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ax2.plot(fpr, tpr, color='blue')
    ax2.set_title("ROC Curve")

    return metrics, fig_cm, fig_roc
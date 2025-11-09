import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import joblib
import os
import traceback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# ⚙️ PAGE CONFIGURATION
# =====================================================
st.set_page_config(page_title="AI-Doctor: Cancer Detection", layout="wide")
st.title("🧠 AI-Doctor: Smart Cancer Prediction System")

try:
    st.write("✅ Streamlit app started successfully.")
except Exception as e:
    st.error(f"❌ Error while rendering: {e}")
    st.text(traceback.format_exc())


# =====================================================
# ⚙️ LOAD MODELS SAFELY
# =====================================================
@st.cache_resource
def load_models():
    img_model_path = "models/image_model_best.h5"
    tab_model_path = "models/tabular_model.pkl"  # or ensemble_tabular.joblib

    img_model, tab_model = None, None

    try:
        if os.path.exists(img_model_path):
            img_model = tf.keras.models.load_model(img_model_path, compile=False)
            st.success("🩻 Image model loaded successfully.")
        else:
            st.warning("⚠️ Image model file not found.")

        if os.path.exists(tab_model_path):
            obj = joblib.load(tab_model_path)
            tab_model = obj.get("model", obj) if isinstance(obj, dict) else obj
            st.success("📊 Tabular model loaded successfully.")
        else:
            st.warning("⚠️ Tabular model file not found.")
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        st.text(traceback.format_exc())

    return img_model, tab_model


img_model, tab_model = load_models()


# =====================================================
# 🧩 GRAD-CAM HEATMAP GENERATION
# =====================================================
def get_gradcam_heatmap(img_tensor, model):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if not last_conv_layer:
        st.warning("⚠️ No Conv2D layer found for Grad-CAM.")
        return np.zeros((224, 224))

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap.numpy()


# =====================================================
# 🩺 IMAGE PREDICTION FUNCTION
# =====================================================
def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_tensor = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_tensor)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    label = "Malignant" if class_idx == 1 else "Benign"

    # Grad-CAM
    heatmap = get_gradcam_heatmap(img_tensor, model)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_rgb = (img_array).astype(np.uint8)
    overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_color, 0.4, 0)

    return label, confidence, preds[0], overlay


# =====================================================
# 📊 TABULAR PREDICTION FUNCTION
# =====================================================
def predict_tabular(model, df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    preds = model.predict(df)
    preds = np.where(preds == 1, "Malignant", "Benign")
    return pd.DataFrame({"Prediction": preds})


# =====================================================
# 📈 MODEL EVALUATION METRICS
# =====================================================
def evaluate_tabular_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_pred)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    return acc, prec, rec, f1, roc


# =====================================================
# 🖥️ MAIN APP UI
# =====================================================
tab_overview, tab_tabular, tab_image, tab_eval = st.tabs([
    "🏠 Overview",
    "🧾 Tabular Data (Clinical Features)",
    "🩻 Image Diagnosis (X-ray/MRI/CT)",
    "📈 Model Evaluation"
])

# 🏠 Overview Tab
with tab_overview:
    st.header("🏥 Welcome to AI-Doctor")
    st.markdown("""
    ### 🧠 AI-Doctor: Smart Cancer Prediction System

📌 Overview

AI-Doctor is an intelligent healthcare assistant that predicts cancer risk using both clinical tabular data and medical images (like X-ray / MRI / histopathology).
It leverages Machine Learning and Deep Learning (CNNs) to assist in early diagnosis — providing both numerical risk scores and visual explanations using Grad-CAM heatmaps.

This system combines:
	•	🧾 Tabular Data Prediction — based on patient clinical features
	•	🩻 Image Diagnosis — based on medical image scans
	•	📊 Evaluation Metrics & Graphs — ROC, AUC, F1, Precision, Recall
	•	🧠 Confidence Visualization — Grad-CAM showing regions influencing prediction


 System Architecture
 📂 AI-Doctor/
│
├── 📁 datasets/                 # Organized train/val/test medical images
│   ├── train/
│   ├── val/
│   └── test/
│
├── 📁 models/                   # Saved models
│   ├── ensemble_tabular.joblib
│   └── image_model_confident.h5
│
├── 📁 scripts/
│   ├── train_image_confident.py # EfficientNetB0 image model training
│   ├── split_dataset.py         # Splits dataset into train/val/test
│   ├── evaluation.py            # Model performance (ROC, F1, AUC)
│
├── 📄 streamlit_app.py          # Main Streamlit web interface
├── 📄 utilities.py              # Data loading, cleaning, scaling utilities
├── 📄 inference.py              # Prediction logic for tabular data
├── 📄 app.py                    # Flask backend (optional REST API)
├── 📄 data.csv                  # Clinical dataset (for tabular model)
└── 📄 README.md                 # Project overview and documentation
🧠 How It Works

🩺 1. Tabular Cancer Prediction
	•	Input: Clinical features (radius_mean, texture_mean, etc.) from data.csv
	•	Preprocessing: Missing value removal, feature scaling, label encoding
	•	Model: Ensemble (RandomForest + LogisticRegression + XGBoost)
	•	Output:
	•	Predicted Label → Benign / Malignant
	•	Confidence Score
	•	Evaluation Metrics: Accuracy, F1, ROC, AUC, Precision, Recall

🧬 2. Image-based Cancer Diagnosis
	•	Input: Histopathology / MRI / X-ray image
	•	Model: EfficientNetB0 (transfer learning)
	•	Training:
	•	10k samples/class (for fast training)
	•	Dropout = 0.15 for higher confidence
	•	Augmentations → rotation, flipping, shifting
	•	Output:
	•	Predicted Label → Benign / Malignant
	•	Confidence Probability
	•	Grad-CAM Heatmap showing focus regions

📊 3. Evaluation Metrics

The system evaluates the trained models using:
	•	Confusion Matrix
	•	ROC & AUC Curve
	•	Precision-Recall Curve
	•	F1-Score, Accuracy, Sensitivity, Specificity

These help in comparing performance across models.

🚀 Usage Guide

1️⃣ Setup Environment
 
 bash

cd cancer_prediction_final
python3 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt

2️⃣ Prepare Dataset
unzip archive.zip -d dataset
python split_dataset.py

3️⃣ Train Models

a. Train Image Model (EfficientNet)
python train_image_confident.py

b. Train Tabular Model
python -c "from utilities import load_tabular, train_and_save; df=load_tabular('data.csv'); train_and_save(df)"


4️⃣ Launch Streamlit App

streamlit run streamlit_app.pyThen open in your browser:
👉 http://localhost:8501￼

⸻

🧪 Outputs

🩻 Image Diagnosis Example
	•	Prediction: Malignant
	•	Confidence: 96.7%
	•	Grad-CAM: Highlighted cancer-affected region

📊 Tabular Evaluation

Metric.  Score
Accuracy 0.97
Precision 0.96
Recall.   0.95
F1-Score  0.96
AUC  0.982

🏁 Key Highlights

✅ Dual-Mode Prediction — Tabular + Image
✅ Transfer Learning with EfficientNetB0
✅ Grad-CAM interpretability for explainable AI
✅ ROC, F1, AUC evaluation metrics
✅ Interactive Streamlit Dashboard
✅ Scalable & modular ML pipeline


📚 Future Scope
	•	Integration with real-time hospital data
	•	Support for multi-class cancer detection
	•	Deployment as an API / mobile app interface
	•	Explainable AI reports for doctors

**

    Upload your data or images in the tabs above to predict cancer risk.
    """)

# 🧾 Tabular Data Tab
with tab_tabular:
    st.header("📊 Cancer Risk Prediction (Tabular Data)")

    uploaded_csv = st.file_uploader("Upload your clinical dataset (CSV)", type=["csv"])
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("Predict Cancer Risk"):
            if tab_model:
                # Drop irrelevant columns automatically
                drop_cols = ['id', 'diagnosis', 'target', 'Unnamed: 32']
                df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
                df_clean = df_clean.select_dtypes(include=['number'])

                st.info(f"🧩 Using {df_clean.shape[1]} features for prediction.")

                try:
                    preds = predict_tabular(tab_model, df_clean)
                    st.success("✅ Predictions Complete!")

                    def highlight_row(row):
                        color = '#2ecc71' if row['Prediction'] == 'Benign' else '#e74c3c'
                        return ['background-color: {}'.format(color)] * len(row)

                    st.dataframe(preds.style.apply(highlight_row, axis=1))
                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")
                    st.text(traceback.format_exc())
            else:
                st.error("⚠️ Tabular model not found. Train it first using `train_and_save()`.")

# 🩻 Image Diagnosis Tab
with tab_image:
    st.header("🩺 Upload an X-ray / MRI / Histopathology Image")

    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        img_path = f"temp_{uploaded_img.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded_img.read())

        st.image(uploaded_img, caption="Uploaded Image", width='stretch')

        if st.button("Analyze Image"):
            if img_model:
                label, conf, probs, heatmap_img = predict_image(img_path, img_model)

                st.subheader(f"🧠 Prediction: {label}")
                st.progress(float(conf))
                st.write(f"Confidence: {conf*100:.2f}%")

                st.bar_chart(pd.DataFrame({
                    "Probability": probs
                }, index=["Benign", "Malignant"]))

                st.subheader("🔥 Model Attention (Grad-CAM)")
                st.image(heatmap_img, caption="Regions Influencing Prediction", use_container_width=True)
            else:
                st.error("⚠️ Image model not found. Please train it first.")

# 📈 Evaluation Tab
# 📈 Evaluation Tab
# 📈 Evaluation Tab
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve

with tab_eval:
    st.header("📊 Evaluate Tabular Model Performance")

    uploaded_eval = st.file_uploader("Upload Test Data (CSV with target column)", type=["csv"])
    target_col = st.text_input("Enter target column name (e.g., diagnosis):")

    if uploaded_eval is not None and target_col:
        try:
            eval_df = pd.read_csv(uploaded_eval)
            eval_df = eval_df.loc[:, ~eval_df.columns.str.contains('^Unnamed')]
            drop_cols = ['id', 'Unnamed: 32']
            eval_df = eval_df.drop(columns=[c for c in drop_cols if c in eval_df.columns], errors='ignore')

            X_test = eval_df.drop(columns=[target_col])
            y_test = eval_df[target_col].replace({'M': 1, 'B': 0}).values

            if tab_model:
                # Predictions
                y_pred = tab_model.predict(X_test)
                if y_pred.ndim > 1:
                    y_pred = np.argmax(y_pred, axis=1)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc = roc_auc_score(y_test, y_pred)

                st.success("✅ Evaluation Complete!")

                # 🔹 Metric Gauges
                st.subheader("🎯 Performance Gauges")
                col1, col2, col3 = st.columns(3)
                col4, col5 = st.columns(2)

                def gauge_plot(title, value, color):
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=value * 100,
                        title={'text': title, 'font': {'size': 22}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 50], 'color': "#f8d7da"},
                                {'range': [50, 80], 'color': "#fff3cd"},
                                {'range': [80, 100], 'color': "#d4edda"}
                            ],
                            'threshold': {'line': {'color': "black", 'width': 4}, 'value': value * 100}
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(t=40, b=0, l=0, r=0))
                    return fig

                col1.plotly_chart(gauge_plot("Accuracy", acc, "#3498db"), use_container_width=True)
                col2.plotly_chart(gauge_plot("Precision", prec, "#9b59b6"), use_container_width=True)
                col3.plotly_chart(gauge_plot("Recall", rec, "#27ae60"), use_container_width=True)
                col4.plotly_chart(gauge_plot("F1 Score", f1, "#e67e22"), use_container_width=True)
                col5.plotly_chart(gauge_plot("ROC-AUC", roc, "#c0392b"), use_container_width=True)

                # --- Metric Table
                st.subheader("📊 Metric Summary")
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                    "Score": [acc, prec, rec, f1, roc]
                })
                st.dataframe(metrics_df.style.format({"Score": "{:.3f}"}))

                # --- Confusion Matrix ---
                st.subheader("🧮 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
                            xticklabels=['Benign', 'Malignant'],
                            yticklabels=['Benign', 'Malignant'])
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                st.pyplot(fig)

                # --- ROC Curve ---
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color="red", lw=2, label=f"ROC curve (AUC = {roc:.3f})")
                ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                st.pyplot(fig)

                # --- Precision-Recall Curve ---
                precision, recall, _ = precision_recall_curve(y_test, y_pred)
                fig, ax = plt.subplots()
                ax.plot(recall, precision, color="blue", lw=2)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curve")
                st.pyplot(fig)

                # --- Feature Importance ---
                st.subheader("🧩 Feature Importance (if available)")
                if hasattr(tab_model, "feature_importances_"):
                    importance = pd.DataFrame({
                        "Feature": X_test.columns,
                        "Importance": tab_model.feature_importances_
                    }).sort_values(by="Importance", ascending=False).head(10)

                    fig, ax = plt.subplots()
                    sns.barplot(x="Importance", y="Feature", data=importance, ax=ax, palette="crest")
                    ax.set_title("Top 10 Important Features")
                    st.pyplot(fig)
                else:
                    st.info("ℹ️ Feature importance not available for this model type.")

            else:
                st.error("⚠️ Tabular model not loaded.")

        except Exception as e:
            st.error(f"⚠️ Evaluation failed: {e}")
            st.text(traceback.format_exc())
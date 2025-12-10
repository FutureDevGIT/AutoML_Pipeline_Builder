# automl_pipeline_builder.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AutoML Pipeline Builder", layout="wide")
st.title("🧠 AutoML Pipeline Builder")
st.markdown("Upload a classification dataset, choose a model, and let the pipeline handle everything from preprocessing to evaluation.")

# Sidebar
st.sidebar.header("📁 Upload Your Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

model_choice = st.sidebar.selectbox("🔍 Choose Model", [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "K-Nearest Neighbors",
    "Support Vector Machine"
])

if file:
    df = pd.read_csv(file)
    st.subheader("🔎 Preview of Uploaded Data")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select Target Column", df.columns)

    if target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Encode categorical features
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Encode target if needed
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model selection
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif model_choice == "K-Nearest Neighbors":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_choice == "Support Vector Machine":
            model = SVC(probability=True)

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # -----------------------------
        #   SAFE PROBABILITY LOGIC
        # -----------------------------
        proba = model.predict_proba(X_test)
        unique_classes = np.unique(y_test)

        st.subheader("📈 ROC AUC Score")

        if len(unique_classes) == 2:
            # Binary classification
            y_proba = proba[:, 1]
            roc_score = roc_auc_score(y_test, y_proba)
            st.success(f"ROC AUC (Binary): {roc_score:.4f}")

        else:
            # Multiclass classification
            try:
                roc_score = roc_auc_score(
                    y_test,
                    proba,
                    multi_class="ovr",
                    average="macro"
                )
                st.success(f"ROC AUC (Multiclass OVR/Macro): {roc_score:.4f}")
                st.info("ℹ️ Multiclass dataset detected — using One-vs-Rest Macro AUC.")
            except Exception as e:
                st.error(f"ROC AUC could not be computed for multiclass: {e}")

        # -------------------------------------
        # Classification Report
        # -------------------------------------
        st.subheader("📊 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # -------------------------------------
        # Confusion Matrix
        # -------------------------------------
        st.subheader("🧩 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # -------------------------------------
        # Export Model + Scaler
        # -------------------------------------
        if st.button("📤 Export Trained Model"):
            joblib.dump(model, "trained_model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(X.columns.tolist(), "feature_order.pkl")
            st.success("Model, scaler, and feature order exported successfully!")

        # -------------------------------------
        # Prediction on new uploaded data
        # -------------------------------------
        st.subheader("📉 Predict on New Data")
        new_file = st.file_uploader("Upload New Data CSV (same columns)", type=["csv"], key="new_data")

        if new_file:
            new_data = pd.read_csv(new_file)
            st.write("🔍 New Data Preview:")
            st.dataframe(new_data.head())

            try:
                feature_order = joblib.load("feature_order.pkl")
                new_data = new_data[feature_order]

                scaler = joblib.load("scaler.pkl")
                model = joblib.load("trained_model.pkl")

                new_scaled = scaler.transform(new_data)
                predictions = model.predict(new_scaled)

                st.write("✅ Predictions:")
                st.write(predictions)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

else:
    st.warning("Please upload a CSV file to begin.")

# 🧠 AutoML Pipeline Builder

A user-friendly Streamlit application that allows anyone to train and evaluate machine learning models on **any classification dataset** — no coding required.

---
## 🖼️ App Preview
![Streamlit App Screenshot](screenshot.png)
---

## 🚀 Features

- 🔍 1. Data Upload & Preview
  - Upload CSV datasets
  - Automatically detect categorical columns
  - Preview first 5 rows

- 📤 **Export trained model** (model, scaler, and feature order)
- 📉 **Upload your own test data** and get real-time predictions

---

- 🧪 ⚙️ 2. Automatic Preprocessing
  - Label encodes target
  - Encodes categorical features
  - Scales numerical columns
  - Stratified train-test split
  - Returns all artifacts (encoders, scaler, feature order)

---

- 🤖 3. Model Selection & Training
  - Models supported:
  - Logistic Regression
  - Random Forest
  - XGBoost (binary & multiclass)
  - K-Nearest Neighbors
  - Support Vector Machine

---

- 📊 4. Evaluation Metrics
  - Supports binary and multiclass classification.
  - Includes:
  - Classification Report
  - Confusion Matrix
  - ROC AUC
  - ROC Curve Plot
  - Precision–Recall Curve Plot

---

- 📈 5. Visualizations
  - Heatmap of confusion matrix
  - ROC curve
  - PR curve
  - Feature importance (RF & XGB only)

---

- 📤 6. Export Model Artifacts
  - Saves:
    ```
    trained_model.pkl
    scaler.pkl
    feature_order.pkl
    target_encoder.pkl
    ```

---

- 📉 7. Predict on New Data
  - Upload new CSV
  - Auto-align columns
  - Apply saved preprocessing
  - Run inference
  - Decode predictions using saved LabelEncoder

---

## ▶️ How to Run Locally

- 1. Clone Repo
```
git clone https://github.com/yourusername/automl_app.git
cd automl_app
```

- 2. Install Dependencies
```
pip install -r requirements.txt
```

- 3. Run Streamlit App
```
streamlit run app.py
```
---
## 🌐 Deploy on Streamlit Community Cloud
- Push the project to GitHub
- Go to https://streamlit.io/cloud
- Click New App
- Choose app.py as entry file
- Deploy 🎉

- The app will run fully on the cloud — including training, exporting, and predictions.
---
## 🛡 Contributing
- Pull requests are welcome!
- For major changes, please open an issue first.
---

## 📚 Useful Concepts Covered
- SMOTE (Synthetic Minority Over-sampling)
- Multi-model comparison
- Feature scaling and label encoding
- VotingClassifier (soft voting)
- UI deployment using Streamlit

## 📜 License
- MIT © 2025 Mayank Raval
---

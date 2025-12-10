# automl_pipeline_builder.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score
)
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ---- Streamlit configuration ----
st.set_page_config(page_title="AutoML Pipeline Builder — Pro", layout="wide")
st.title("🧠 AutoML Pipeline Builder — Professional")
st.markdown(
    """
Upload a classification dataset, choose options and let the app train, evaluate, and export a production-ready model + report.
"""
)

# ---- Sidebar: global controls ----
st.sidebar.header("Settings")
file = st.sidebar.file_uploader("Upload CSV (classification)", type=["csv"])
test_size = st.sidebar.slider("Test set size (%)", 10, 50, 30)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)
train_all_models = st.sidebar.checkbox("Train all models (leaderboard)", value=True)
show_curves = st.sidebar.checkbox("Show ROC / PR curves", value=True)
export_dir = "."  # in Streamlit Cloud artifacts are visible in workspace

# Model list (order)
ALL_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", tree_method="hist"),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(probability=True)
}

# ---- Helper functions ----
def safe_target_encode(y_series):
    """Label-encode target safely and return (encoded, encoder)."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.astype(str))
    return y_enc.astype(np.int32), le

def fit_model(model, X_train, y_train):
    """Train model with try/except (some models may raise)."""
    try:
        model.fit(X_train, y_train)
        return model, None
    except Exception as e:
        return None, e

def compute_roc_pr(y_test, y_score, classes):
    """
    y_test: original integer labels (shape n_samples,)
    y_score: probability scores (n_samples, n_classes) or (n_samples,) for binary
    classes: list/array of class labels
    Returns dict with ROC and PR plotting data and summary AUCs.
    """
    n_classes = len(classes)
    result = {}

    # Binarize the test labels for multiclass curve plotting
    y_test_binarized = label_binarize(y_test, classes=classes)
    if n_classes == 1:
        # Degenerate case (shouldn't happen)
        return result

    # If binary, y_score might be shape (n,) or (n,2)
    if n_classes == 2:
        if y_score.ndim == 2:
            score_pos = y_score[:, 1]
        else:
            score_pos = y_score
        fpr, tpr, _ = roc_curve(y_test, score_pos)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_test, score_pos)
        ap = average_precision_score(y_test, score_pos)

        result['binary'] = {
            'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
            'precision': precision, 'recall': recall, 'ap': ap
        }
    else:
        # Multiclass: compute OVR ROC for each class + micro-average
        fpr = dict(); tpr = dict(); roc_auc = dict()
        precision = dict(); recall = dict(); average_precision = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test_binarized[:, i], y_score[:, i])

        # micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_binarized.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test_binarized, y_score, average="micro")

        result['multiclass'] = {
            'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
            'precision': precision, 'recall': recall, 'average_precision': average_precision
        }

    return result

def plot_roc(result, classes, title="ROC Curve"):
    fig, ax = plt.subplots(figsize=(7,5))
    if 'binary' in result:
        r = result['binary']
        ax.plot(r['fpr'], r['tpr'], label=f"AUC = {r['roc_auc']:.4f}")
        ax.plot([0,1], [0,1], linestyle='--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
    else:
        r = result['multiclass']
        # plot micro
        ax.plot(r['fpr']['micro'], r['tpr']['micro'], label=f"micro (AUC={r['roc_auc']['micro']:.4f})", linewidth=2)
        # plot per class
        for i, cls in enumerate(classes):
            ax.plot(r['fpr'][i], r['tpr'][i], lw=1, label=f"Class {cls} (AUC={r['roc_auc'][i]:.4f})")
        ax.plot([0,1], [0,1], linestyle='--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize='small')
    fig.tight_layout()
    return fig

def plot_pr(result, classes, title="Precision-Recall Curve"):
    fig, ax = plt.subplots(figsize=(7,5))
    if 'binary' in result:
        r = result['binary']
        ax.plot(r['recall'], r['precision'], label=f"AP = {r['ap']:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
    else:
        r = result['multiclass']
        ax.plot(r['recall']['micro'], r['precision']['micro'], label=f"micro AP={r['average_precision']['micro']:.4f}", linewidth=2)
        for i, cls in enumerate(classes):
            ax.plot(r['recall'][i], r['precision'][i], lw=1, label=f"Class {cls} (AP={r['average_precision'][i]:.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize='small')
    fig.tight_layout()
    return fig

def feature_importance_plot(model, feature_names):
    """
    Returns a matplotlib figure for feature importances/coefs where available.
    Supports RandomForest, XGBoost (feature_importances_), and LogisticRegression (coef_).
    """
    fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names)*0.2)))
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], ax=ax)
        ax.set_title("Feature Importances")
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_).ravel()
        idx = np.argsort(coefs)[::-1]
        sns.barplot(x=coefs[idx], y=np.array(feature_names)[idx], ax=ax)
        ax.set_title("Absolute Coefficients (Logistic Regression)")
    else:
        ax.text(0.5, 0.5, "Feature importances not available for this model", ha='center')
    fig.tight_layout()
    return fig

# ---- Main app flow: Tabs ----
tabs = st.tabs(["Data", "Training", "Evaluation", "Predict", "Export"])
data_tab, train_tab, eval_tab, predict_tab, export_tab = tabs

# ---- DATA TAB ----
with data_tab:
    st.header("1) Data")
    if file is None:
        st.info("Upload a CSV file from the left sidebar to begin.")
    else:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.subheader("Preview")
        st.dataframe(df.head())

        st.subheader("Shape & types")
        st.write("Rows:", df.shape[0], " Columns:", df.shape[1])
        st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))

        with st.expander("Show column sample values (first 5 rows)"):
            st.write(df.head())

        # Target selection
        target_col = st.selectbox("Select target (label) column", df.columns)
        if target_col:
            st.success(f"Target selected: {target_col}")

# ---- TRAINING TAB ----
with train_tab:
    st.header("2) Training")
    if file is None:
        st.info("Upload data first.")
    else:
        st.write("Preparing data and encoding target...")

        # Safe target encoding (before split)
        y_raw = df[target_col]
        y, target_encoder = safe_target_encode(y_raw)
        X = df.drop(columns=[target_col]).copy()

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        feature_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            feature_encoders[col] = le

        st.write(f"Detected categorical features: {categorical_cols}")
        st.write(f"Number of classes: {len(np.unique(y))}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100.0, stratify=y, random_state=int(random_state)
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model selection UI
        st.subheader("Model selection")
        chosen_model_name = st.selectbox("Train a single model (when unchecked, training all models for leaderboard)", ["Logistic Regression","Random Forest","XGBoost","K-Nearest Neighbors","Support Vector Machine"])
        single_only = st.checkbox("Train single selected model only (skip leaderboard)", value=False)
        if single_only:
            st.info("Single-model mode: only the selected model will be trained.")
        # Allow override of train_all_models from sidebar:
        if single_only:
            train_all_models_flag = False
        else:
            train_all_models_flag = train_all_models

        # Prepare set of models to train
        models_to_train = {}
        if train_all_models_flag:
            for name, base_model in ALL_MODELS.items():
                # clone a fresh estimator (safe)
                if name == "XGBoost":
                    num_classes = len(np.unique(y))
                    # configure XGBoost safely
                    m = XGBClassifier(
                        objective="multi:softprob" if num_classes > 2 else "binary:logistic",
                        eval_metric="mlogloss",
                        tree_method="hist",
                        use_label_encoder=False
                    )
                else:
                    m = base_model.__class__(**(base_model.get_params()))
                models_to_train[name] = m
        else:
            # only chosen_model_name
            name = chosen_model_name
            if name == "XGBoost":
                num_classes = len(np.unique(y))
                m = XGBClassifier(
                    objective="multi:softprob" if num_classes > 2 else "binary:logistic",
                    eval_metric="mlogloss",
                    tree_method="hist",
                    use_label_encoder=False
                )
            else:
                base_model = ALL_MODELS[name]
                m = base_model.__class__(**(base_model.get_params()))
            models_to_train[name] = m

        st.write(f"Models to train: {list(models_to_train.keys())}")

        # Train models with progress bar
        progress = st.progress(0)
        results = {}
        total = len(models_to_train)
        i = 0
        for name, model in models_to_train.items():
            i += 1
            st.write(f"Training: **{name}**")
            status_text = st.empty()
            status_text.info("Training... (this can take a few seconds)")

            trained, error = fit_model(model, X_train_scaled, y_train)
            if error is not None:
                status_text.error(f"Training failed for {name}: {error}")
                results[name] = {"model": None, "error": error}
            else:
                status_text.success(f"Trained {name}")
                # Compute predictions and probabilities if possible
                try:
                    y_pred = trained.predict(X_test_scaled)
                except Exception as e:
                    y_pred = None

                proba = None
                try:
                    proba = trained.predict_proba(X_test_scaled)
                except Exception:
                    # Some classifiers might not support predict_proba
                    try:
                        # for SVM or others, try decision_function then convert (only for binary)
                        dfun = trained.decision_function(X_test_scaled)
                        if dfun.ndim == 1:
                            # binary: convert with sigmoid-like
                            proba = np.vstack([1 - (dfun - dfun.min())/(dfun.max()-dfun.min()+1e-9),
                                               (dfun - dfun.min())/(dfun.max()-dfun.min()+1e-9)]).T
                        else:
                            # multiclass decision_function -> softmax-like
                            exp = np.exp(dfun)
                            proba = exp / np.sum(exp, axis=1, keepdims=True)
                    except Exception:
                        proba = None

                acc = accuracy_score(y_test, y_pred) if y_pred is not None else np.nan

                # Compute ROC AUC robustly
                roc_auc = np.nan
                try:
                    if proba is not None:
                        if len(np.unique(y_test)) == 2:
                            # binary
                            roc_auc = roc_auc_score(y_test, proba[:, 1])
                        else:
                            roc_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
                except Exception:
                    roc_auc = np.nan

                results[name] = {
                    "model": trained,
                    "y_pred": y_pred,
                    "proba": proba,
                    "accuracy": acc,
                    "roc_auc": roc_auc,
                    "error": None
                }

            progress.progress(int(i/total*100))
            time.sleep(0.1)

        st.success("Training complete.")
        progress.empty()

        # Save train/test objects to session_state for evaluation and predict tabs
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['X_train_scaled'] = X_train_scaled
        st.session_state['X_test_scaled'] = X_test_scaled
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['feature_names'] = X.columns.tolist()
        st.session_state['feature_encoders'] = feature_encoders
        st.session_state['scaler'] = scaler
        st.session_state['results'] = results
        st.session_state['target_encoder'] = target_encoder
        st.session_state['raw_df'] = df

        # Leaderboard
        st.subheader("Leaderboard")
        leaderboard = []
        for name, r in results.items():
            leaderboard.append({
                "model": name,
                "accuracy": r.get("accuracy", np.nan),
                "roc_auc": r.get("roc_auc", np.nan),
                "status": "OK" if r.get("error") is None else "ERROR"
            })
        lb = pd.DataFrame(leaderboard).sort_values(["roc_auc", "accuracy"], ascending=[False, False])
        st.dataframe(lb.reset_index(drop=True))

        # Best model selection (first OK model with highest roc_auc)
        best_row = lb[lb["status"] == "OK"].head(1)
        if not best_row.empty:
            best_model_name = best_row.iloc[0]["model"]
            st.session_state['best_model_name'] = best_model_name
            st.success(f"Selected best model: {best_model_name}")
        else:
            st.warning("No successful model to select as best.")

# ---- EVALUATION TAB ----
with eval_tab:
    st.header("3) Evaluation")
    if 'results' not in st.session_state:
        st.info("Train models first in the Training tab.")
    else:
        results = st.session_state['results']
        y_test = st.session_state['y_test']
        feature_names = st.session_state['feature_names']

        # Select which model to inspect
        inspect_model_name = st.selectbox("Choose model to inspect", list(results.keys()))
        r = results[inspect_model_name]
        if r["error"] is not None or r["model"] is None:
            st.error(f"Model {inspect_model_name} failed during training: {r['error']}")
        else:
            model = r["model"]
            y_pred = r["y_pred"]
            proba = r["proba"]

            st.subheader("Classification report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("Confusion matrix")
            fig_cm, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig_cm)

            # ROC & PR curves
            if show_curves and proba is not None:
                classes = np.unique(st.session_state['y_train'])
                rocpr = compute_roc_pr(y_test, proba, classes)
                if rocpr:
                    fig_roc = plot_roc(rocpr, classes, title=f"ROC Curve — {inspect_model_name}")
                    fig_pr = plot_pr(rocpr, classes, title=f"Precision-Recall — {inspect_model_name}")
                    st.subheader("ROC Curve")
                    st.pyplot(fig_roc)
                    st.subheader("Precision-Recall Curve")
                    st.pyplot(fig_pr)
                else:
                    st.info("ROC/PR could not be computed for this model.")
            else:
                if proba is None:
                    st.info("Probability scores not available for this model, ROC/PR curves unavailable.")

            # Feature importances
            st.subheader("Feature importances / coefficients")
            fig_imp = feature_importance_plot(model, feature_names)
            st.pyplot(fig_imp)

# ---- PREDICT TAB ----
with predict_tab:
    st.header("4) Predict")
    if 'best_model_name' not in st.session_state:
        st.info("Train models first and let the app select the best model (Training tab).")
    else:
        best_name = st.session_state['best_model_name']
        st.write(f"Using **{best_name}** as the production model")
        best_model = st.session_state['results'][best_name]['model']
        scaler = st.session_state['scaler']
        feature_order = st.session_state['feature_names']
        target_encoder = st.session_state['target_encoder']
        feature_encoders = st.session_state['feature_encoders']

        new_file = st.file_uploader("Upload new data for prediction (CSV)", type=["csv"], key="predict_new")
        if new_file:
            new_df = pd.read_csv(new_file)
            st.write("Preview:")
            st.dataframe(new_df.head())

            try:
                # Ensure same columns
                new_df = new_df[feature_order]
                # encode categorical features using stored encoders if available
                for col, enc in feature_encoders.items():
                    if col in new_df.columns:
                        new_df[col] = enc.transform(new_df[col].astype(str))
                new_scaled = scaler.transform(new_df)
                preds = best_model.predict(new_scaled)
                decoded = target_encoder.inverse_transform(preds)
                st.subheader("Predictions (decoded labels)")
                st.write(decoded)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---- EXPORT TAB ----
with export_tab:
    st.header("5) Export")
    if 'results' not in st.session_state:
        st.info("Train models first.")
    else:
        results = st.session_state['results']
        scaler = st.session_state['scaler']
        feature_names = st.session_state['feature_names']
        target_encoder = st.session_state['target_encoder']
        raw_df = st.session_state['raw_df']

        # Which model to export?
        export_choice = st.selectbox("Choose model to export", list(results.keys()), index=0)
        r = results[export_choice]
        if r["model"] is None:
            st.error("Selected model not available to export.")
        else:
            model = r["model"]

            # Export artifacts button
            if st.button("Export model, scaler, feature order, target encoder"):
                try:
                    joblib.dump(model, f"{export_dir}/trained_model_{export_choice}.pkl")
                    joblib.dump(scaler, f"{export_dir}/scaler.pkl")
                    joblib.dump(feature_names, f"{export_dir}/feature_order.pkl")
                    joblib.dump(target_encoder, f"{export_dir}/target_encoder.pkl")
                    st.success("Exported artifacts to app workspace.")
                except Exception as e:
                    st.error(f"Export failed: {e}")

            # Create PDF report
            if st.button("Generate PDF report (multi-page)"):
                try:
                    # Build PDF in-memory
                    buffer = io.BytesIO()
                    with PdfPages(buffer) as pdf:
                        # Cover page (text)
                        fig, ax = plt.subplots(figsize=(8.27, 11.69))
                        ax.axis('off')
                        ax.text(0.5, 0.8, "AutoML Pipeline Builder — Report", ha='center', fontsize=20, weight='bold')
                        ax.text(0.5, 0.72, f"Exported model: {export_choice}", ha='center', fontsize=12)
                        ax.text(0.5, 0.68, f"Date: {pd.Timestamp.now()}", ha='center', fontsize=10)
                        ax.text(0.1, 0.5, "Dataset summary:", fontsize=12, weight='bold')
                        ax.text(0.1, 0.46, f"Rows: {raw_df.shape[0]}  Columns: {raw_df.shape[1]}", fontsize=10)
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

                        # Add leaderboard table as figure
                        lb_rows = []
                        for name, res in results.items():
                            lb_rows.append({
                                "model": name,
                                "accuracy": res.get("accuracy", np.nan),
                                "roc_auc": res.get("roc_auc", np.nan),
                                "status": "OK" if res.get("error") is None else "ERROR"
                            })
                        lb_df = pd.DataFrame(lb_rows).sort_values(["roc_auc", "accuracy"], ascending=[False, False])
                        fig, ax = plt.subplots(figsize=(8.27, 3 + 0.2*len(lb_df)))
                        ax.axis('off')
                        ax.table(cellText=np.round(lb_df.fillna(0).values, 4),
                                 colLabels=lb_df.columns,
                                 loc='center')
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

                        # For each model include key plots
                        for name, res in results.items():
                            if res.get("model") is None:
                                continue
                            mod = res["model"]
                            proba = res.get("proba")
                            y_test = st.session_state['y_test']
                            # ROC / PR
                            if proba is not None:
                                classes = np.unique(st.session_state['y_train'])
                                rocpr = compute_roc_pr(y_test, proba, classes)
                                if rocpr:
                                    fig_roc = plot_roc(rocpr, classes, title=f"ROC — {name}")
                                    pdf.savefig(fig_roc, bbox_inches='tight')
                                    plt.close(fig_roc)
                                    fig_pr = plot_pr(rocpr, classes, title=f"Precision-Recall — {name}")
                                    pdf.savefig(fig_pr, bbox_inches='tight')
                                    plt.close(fig_pr)
                            # Confusion matrix
                            if res.get("y_pred") is not None:
                                fig_cm, ax = plt.subplots(figsize=(8,5))
                                sns.heatmap(confusion_matrix(y_test, res["y_pred"]), annot=True, fmt='d', cmap='Blues', ax=ax)
                                ax.set_title(f"Confusion Matrix — {name}")
                                pdf.savefig(fig_cm, bbox_inches='tight')
                                plt.close(fig_cm)
                            # Feature importance (if available)
                            fig_imp = feature_importance_plot(mod, feature_names)
                            pdf.savefig(fig_imp, bbox_inches='tight')
                            plt.close(fig_imp)

                    buffer.seek(0)
                    b64 = buffer.read()
                    st.download_button(
                        label="Download PDF report",
                        data=b64,
                        file_name=f"automl_report_{export_choice}.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF generated (download available).")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

            # Quick test export of best model as single file
            if st.button("Export best model as 'production_model.pkl'"):
                try:
                    joblib.dump(model, f"{export_dir}/production_model_{export_choice}.pkl")
                    st.success("Production model exported.")
                except Exception as e:
                    st.error(f"Export failed: {e}")

st.markdown("---")
st.caption("Built with ♥ — AutoML Pipeline Builder. Make sure your dataset is classification and contains at least two label classes.")

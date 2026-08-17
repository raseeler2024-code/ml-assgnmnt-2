"""Interactive Streamlit dashboard for comparing digit classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ml_utils import (
    FEATURE_NAMES,
    MODEL_FILES,
    TARGET_COLUMN,
    calculate_metrics,
    predict_probabilities,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Digit Model Lab", page_icon="🔢", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background: #ffffff; border: 1px solid #fed7aa;
        border-radius: 16px; padding: 0.8rem 1rem; box-shadow: 0 5px 18px rgba(50,35,20,.05);}
      .eyebrow {color:#c2410c; font-weight:800; letter-spacing:.12em; font-size:.78rem;}
      .hero {background:linear-gradient(120deg,#172033,#263653); color:white;
        padding:1.6rem 1.8rem; border-radius:22px; margin-bottom:1.2rem;}
      .hero h1 {margin:0.25rem 0; color:white; font-size:2.35rem;}
      .hero p {margin:.35rem 0 0; color:#d7e0ef; max-width:760px;}
      .note {background:#fff7ed; border-left:4px solid #f97316; padding:.8rem 1rem;
        border-radius:0 12px 12px 0;}
    </style>
    <div class="hero">
      <div class="eyebrow">MACHINE LEARNING ASSIGNMENT 2</div>
      <h1>Digit Model Lab</h1>
      <p>Compare six classifiers on 8 × 8 handwritten-digit features, inspect errors,
      and download row-level predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets() -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = MODEL_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Model assets are missing. Run: python train_models.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    return models, manifest


def read_uploaded_data(uploaded_file: object | None) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.read_csv(ROOT / "test_data.csv")
    return pd.read_csv(uploaded_file)


def validate_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    missing = [column for column in FEATURE_NAMES if column not in frame.columns]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing {len(missing)} required feature columns (for example: {preview}).")

    features = frame.loc[:, FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("Feature columns must contain numeric, non-missing values.")

    target = None
    if TARGET_COLUMN in frame.columns:
        numeric_target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        if numeric_target.isna().any() or not numeric_target.isin(range(10)).all():
            raise ValueError("The target column must contain integer digit labels from 0 to 9.")
        target = numeric_target.astype(int)
    return features, target


try:
    models, manifest = load_assets()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Experiment controls")
    uploaded = st.file_uploader("Upload test data", type=["csv"], help="Use the supplied test_data.csv format.")
    selected_name = st.selectbox("Focus model", list(models))
    st.caption("No upload? The app uses the bundled stratified holdout set.")
    st.divider()
    st.write("**Dataset profile**")
    st.write(f"{manifest['dataset_rows']:,} rows • {manifest['feature_count']} features • 10 classes")
    st.write(f"Fixed split seed: `{manifest['random_state']}`")

try:
    raw_data = read_uploaded_data(uploaded)
    features, target = validate_data(raw_data)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"The CSV could not be evaluated: {error}")
    st.stop()

source_label = uploaded.name if uploaded is not None else "bundled test_data.csv"
st.markdown(f"<div class='note'><b>Active data:</b> {source_label} • {len(raw_data):,} rows</div>", unsafe_allow_html=True)

selected_model = models[selected_name]
selected_predictions = np.asarray(selected_model.predict(features))
selected_probabilities = predict_probabilities(selected_model, features)

if target is not None:
    st.subheader("Model comparison")
    comparison_rows = [
        {"ML Model Name": name, **calculate_metrics(model, features, target)}
        for name, model in models.items()
    ]
    comparison = pd.DataFrame(comparison_rows).sort_values("F1", ascending=False, ignore_index=True)
    styled_comparison = comparison.style.format(
        {metric: "{:.4f}" for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]},
        na_rep="N/A",
    ).background_gradient(subset=["F1"], cmap="Oranges")
    st.dataframe(styled_comparison, width="stretch", hide_index=True)

    selected_metrics = calculate_metrics(selected_model, features, target)
    metric_columns = st.columns(6)
    for column, metric in zip(metric_columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        value = selected_metrics[metric]
        column.metric(metric, "N/A" if np.isnan(value) else f"{value:.4f}")

    st.subheader(f"Error analysis · {selected_name}")
    chart_column, report_column = st.columns([1.05, 1])
    with chart_column:
        matrix = confusion_matrix(target, selected_predictions, labels=range(10))
        figure, axis = plt.subplots(figsize=(7.3, 5.8))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=axis)
        axis.set_xlabel("Predicted digit")
        axis.set_ylabel("Actual digit")
        axis.set_title("Confusion matrix")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with report_column:
        report = classification_report(
            target,
            selected_predictions,
            labels=list(range(10)),
            output_dict=True,
            zero_division=0,
        )
        report_frame = pd.DataFrame(report).T.loc[[str(i) for i in range(10)], ["precision", "recall", "f1-score", "support"]]
        st.write("**Per-class classification report**")
        st.dataframe(
            report_frame.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}),
            width="stretch",
        )
else:
    st.info("This file has no target column, so evaluation metrics are unavailable. Predictions are still shown below.")

st.subheader("Prediction explorer")
predictions = raw_data.copy()
predictions["predicted_digit"] = selected_predictions
predictions["confidence"] = selected_probabilities.max(axis=1)
if target is not None:
    predictions["correct"] = target.to_numpy() == selected_predictions

display_columns = [column for column in [TARGET_COLUMN, "predicted_digit", "confidence", "correct"] if column in predictions]
st.dataframe(
    predictions.loc[:, display_columns].style.format({"confidence": "{:.1%}"}),
    width="stretch",
    hide_index=True,
)
st.download_button(
    "Download predictions as CSV",
    data=predictions.to_csv(index=False).encode("utf-8"),
    file_name="digit_predictions.csv",
    mime="text/csv",
)

with st.expander("Preview input data and schema"):
    st.dataframe(raw_data.head(20), width="stretch", hide_index=True)
    st.caption(f"Expected columns: 64 pixel features (`pixel_0_0` … `pixel_7_7`) and optional `{TARGET_COLUMN}`.")

st.caption("Metrics use weighted multiclass averaging. AUC uses weighted one-vs-rest probabilities.")

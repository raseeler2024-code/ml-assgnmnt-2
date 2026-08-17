"""Train, evaluate, and persist every classifier used by the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from ml_utils import (
    FEATURE_NAMES,
    MODEL_FILES,
    RANDOM_STATE,
    TARGET_COLUMN,
    build_models,
    calculate_metrics,
    load_assignment_dataset,
    predict_probabilities,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    features, target = load_assignment_dataset()
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    test_data = test_x.copy()
    test_data[TARGET_COLUMN] = test_y.to_numpy()
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    result_rows: list[dict[str, object]] = []
    prediction_frame = pd.DataFrame({"actual": test_y.to_numpy()})

    for model_name, model in build_models().items():
        print(f"Training {model_name}...")
        model.fit(train_x, train_y)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[model_name])

        result_rows.append({"ML Model Name": model_name, **calculate_metrics(model, test_x, test_y)})
        prediction_frame[f"{model_name} prediction"] = model.predict(test_x)
        prediction_frame[f"{model_name} confidence"] = predict_probabilities(model, test_x).max(axis=1)

    results = pd.DataFrame(result_rows).sort_values("F1", ascending=False, ignore_index=True)
    results.to_csv(REPORT_DIR / "model_results.csv", index=False, float_format="%.6f")
    prediction_frame.to_csv(REPORT_DIR / "test_predictions.csv", index=False, float_format="%.6f")

    manifest = {
        "dataset": "scikit-learn load_digits copy of the UCI Optical Recognition dataset",
        "source_url": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        "dataset_rows": int(len(features)),
        "feature_count": int(len(FEATURE_NAMES)),
        "training_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "target_column": TARGET_COLUMN,
        "classes": sorted(int(value) for value in target.unique()),
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "scikit_learn_version": sklearn.__version__,
        "models": MODEL_FILES,
    }
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\nEvaluation results")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved {len(MODEL_FILES)} models and {len(test_data)} test rows in {ROOT}")


if __name__ == "__main__":
    main()

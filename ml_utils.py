"""Shared data, model, and evaluation helpers for the digit classifier."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "target"
FEATURE_NAMES = [f"pixel_{row}_{column}" for row in range(8) for column in range(8)]

MODEL_FILES = OrderedDict(
    {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "k-Nearest Neighbors": "knn.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "Random Forest (Ensemble)": "random_forest.joblib",
        "Support Vector Machine (Additional)": "support_vector_machine.joblib",
    }
)


def load_assignment_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the local scikit-learn copy of the UCI optical-digits test set."""

    digits = load_digits()
    features = pd.DataFrame(digits.data, columns=FEATURE_NAMES)
    target = pd.Series(digits.target.astype(int), name=TARGET_COLUMN)
    return features, target


def build_models() -> OrderedDict[str, object]:
    """Return deterministic estimators for all required and additional models."""

    return OrderedDict(
        {
            "Logistic Regression": Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=3_000,
                            solver="lbfgs",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
            ),
            "k-Nearest Neighbors": Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance")),
                ]
            ),
            "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-9),
            "Random Forest (Ensemble)": RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=1,
                max_features="sqrt",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            "Support Vector Machine (Additional)": CalibratedClassifierCV(
                estimator=Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "classifier",
                            SVC(C=5.0, kernel="rbf", random_state=RANDOM_STATE),
                        ),
                    ]
                ),
                method="sigmoid",
                cv=5,
                ensemble=False,
            ),
        }
    )


def predict_probabilities(model: object, features: pd.DataFrame) -> np.ndarray:
    """Return class probabilities and fail clearly for an incompatible estimator."""

    if not hasattr(model, "predict_proba"):
        raise TypeError("The selected model does not expose predict_proba().")
    return np.asarray(model.predict_proba(features))


def calculate_metrics(model: object, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Calculate all rubric metrics using weighted multiclass averaging."""

    predictions = np.asarray(model.predict(features))
    probabilities = predict_probabilities(model, features)

    try:
        auc = roc_auc_score(
            target,
            probabilities,
            labels=np.asarray(model.classes_),
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        # AUC is undefined when an uploaded test set omits one or more classes.
        auc = float("nan")

    return {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": float(auc),
        "Precision": float(
            precision_score(target, predictions, average="weighted", zero_division=0)
        ),
        "Recall": float(recall_score(target, predictions, average="weighted", zero_division=0)),
        "F1": float(f1_score(target, predictions, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predictions)),
    }

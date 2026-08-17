"""Fast integrity tests for generated data and persisted classifiers."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_utils import FEATURE_NAMES, MODEL_FILES, TARGET_COLUMN, calculate_metrics  # noqa: E402


class ProjectIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data = pd.read_csv(ROOT / "test_data.csv")
        cls.features = cls.test_data.loc[:, FEATURE_NAMES]
        cls.target = cls.test_data[TARGET_COLUMN]

    def test_test_data_meets_schema(self) -> None:
        self.assertGreaterEqual(len(self.test_data), 100)
        self.assertEqual(len(FEATURE_NAMES), 64)
        self.assertEqual(self.features.isna().sum().sum(), 0)
        self.assertEqual(set(self.target.unique()), set(range(10)))

    def test_manifest_and_model_files(self) -> None:
        manifest = json.loads((ROOT / "model" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["feature_count"], 64)
        self.assertEqual(manifest["dataset_rows"], 1797)
        for filename in MODEL_FILES.values():
            self.assertTrue((ROOT / "model" / filename).is_file(), filename)

    def test_models_predict_and_report_all_metrics(self) -> None:
        expected = {"Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"}
        for name, filename in MODEL_FILES.items():
            with self.subTest(model=name):
                model = joblib.load(ROOT / "model" / filename)
                predictions = model.predict(self.features)
                self.assertEqual(len(predictions), len(self.test_data))
                metrics = calculate_metrics(model, self.features, self.target)
                self.assertEqual(set(metrics), expected)
                self.assertTrue(all(0.0 <= metrics[key] <= 1.0 for key in expected))


if __name__ == "__main__":
    unittest.main()

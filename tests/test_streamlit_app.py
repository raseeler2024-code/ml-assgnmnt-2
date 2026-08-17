"""Headless smoke test for the deployed Streamlit experience."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class StreamlitSmokeTests(unittest.TestCase):
    def test_default_data_and_model_selection_render(self) -> None:
        app = AppTest.from_file(ROOT / "app.py", default_timeout=60)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox), 1)
        self.assertEqual(len(app.metric), 6)
        self.assertGreaterEqual(len(app.dataframe), 3)

        app.selectbox[0].select("Gaussian Naive Bayes").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 6)


if __name__ == "__main__":
    unittest.main()

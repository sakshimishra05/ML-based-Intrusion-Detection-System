import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.inference import run_detection
from src.preprocessor import preprocess_csv

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
NORMAL_CSV = os.path.join(SAMPLE_DIR, "normal_traffic.csv")
ATTACK_CSV = os.path.join(SAMPLE_DIR, "attack_traffic.csv")


def test_app_imports():
    import app  # noqa: F401
    print("PASS: app.py imports successfully")


def test_normal_csv_through_run_detection():
    results, X, df_selected = run_detection(NORMAL_CSV)
    assert results is not None
    assert len(results) == len(X)
    print("PASS: normal CSV processed through run_detection()")


def test_attack_csv_through_run_detection():
    results, X, df_selected = run_detection(ATTACK_CSV)
    assert results is not None
    assert len(results) == len(X)
    print("PASS: attack CSV processed through run_detection()")


def test_run_detection_returns_dataframe():
    results, X, df_selected = run_detection(NORMAL_CSV)
    assert isinstance(results, pd.DataFrame)
    assert "row_id" in results.columns
    assert "final_prediction" in results.columns
    print("PASS: run_detection returns DataFrame with required columns")


if __name__ == "__main__":
    test_app_imports()
    test_normal_csv_through_run_detection()
    test_attack_csv_through_run_detection()
    test_run_detection_returns_dataframe()
    print("\nAll app integration tests passed.")
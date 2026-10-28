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
    binary_attack_count = int((results["binary_prediction"] == "ATTACK").sum())
    binary_normal_count = int((results["binary_prediction"] == "NORMAL").sum())
    assert len(results) == 200
    assert binary_attack_count == 5
    assert binary_normal_count == 195
    from app import get_batch_detection_result
    detected, text = get_batch_detection_result(results["binary_prediction"])
    assert not detected
    assert text == (
        "2.5% of traffic records were classified as attack traffic, "
        "below the 5% alert threshold."
    )
    print("PASS: normal CSV processed through run_detection()")


def test_attack_csv_through_run_detection():
    results, X, df_selected = run_detection(ATTACK_CSV)
    assert results is not None
    assert len(results) == len(X)
    binary_attack_count = int((results["binary_prediction"] == "ATTACK").sum())
    binary_normal_count = int((results["binary_prediction"] == "NORMAL").sum())
    assert len(results) == 200
    assert binary_attack_count == 200
    assert binary_normal_count == 0
    from app import get_batch_detection_result
    detected, text = get_batch_detection_result(results["binary_prediction"])
    assert detected
    assert text == "100.0% of traffic records were classified as attack traffic."
    print("PASS: attack CSV processed through run_detection()")


def test_run_detection_returns_dataframe():
    results, X, df_selected = run_detection(NORMAL_CSV)
    assert isinstance(results, pd.DataFrame)
    assert "row_id" in results.columns
    assert "final_prediction" in results.columns
    print("PASS: run_detection returns DataFrame with required columns")


def test_batch_detection_result_threshold():
    from app import get_batch_detection_result

    below_threshold = pd.Series(["ATTACK"] * 5 + ["NORMAL"] * 195)
    detected, text = get_batch_detection_result(below_threshold)
    assert not detected
    assert text == (
        "2.5% of traffic records were classified as attack traffic, "
        "below the 5% alert threshold."
    )

    at_threshold = pd.Series(["ATTACK"] * 10 + ["NORMAL"] * 190)
    detected, text = get_batch_detection_result(at_threshold)
    assert detected
    assert text == "5.0% of traffic records were classified as attack traffic."

    all_attack = pd.Series(["ATTACK"] * 200)
    detected, text = get_batch_detection_result(all_attack)
    assert detected
    assert text == "100.0% of traffic records were classified as attack traffic."


if __name__ == "__main__":
    test_app_imports()
    test_normal_csv_through_run_detection()
    test_attack_csv_through_run_detection()
    test_run_detection_returns_dataframe()
    test_batch_detection_result_threshold()
    print("\nAll app integration tests passed.")
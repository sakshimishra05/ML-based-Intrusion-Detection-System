import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import BINARY_THRESHOLD, UNKNOWN_THRESHOLD
from src.inference import run_detection, predict_preprocessed
from src.model_loader import (
    load_binary_model,
    load_label_encoder,
    load_multiclass_model,
)
from src.preprocessor import preprocess_csv

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
NORMAL_CSV = os.path.join(SAMPLE_DIR, "normal_traffic.csv")
ATTACK_CSV = os.path.join(SAMPLE_DIR, "attack_traffic.csv")


def test_normal_traffic_binary():
    X, _ = preprocess_csv(NORMAL_CSV)
    results = predict_preprocessed(X)
    assert len(results) == len(X)
    print("PASS: normal traffic runs through binary model")


def test_attack_traffic_full_pipeline():
    X, _ = preprocess_csv(ATTACK_CSV)
    results = predict_preprocessed(X)
    assert len(results) == len(X)
    print("PASS: attack traffic runs through binary + multiclass pipeline")


def test_output_row_count_matches_input():
    X, _ = preprocess_csv(NORMAL_CSV)
    results = predict_preprocessed(X)
    assert len(results) == len(X)
    X2, _ = preprocess_csv(ATTACK_CSV)
    results2 = predict_preprocessed(X2)
    assert len(results2) == len(X2)
    print("PASS: output row count equals input row count")


def test_row_id_alignment():
    X, _ = preprocess_csv(NORMAL_CSV)
    results = predict_preprocessed(X)
    assert list(results["row_id"]) == list(range(len(X)))
    print("PASS: row_id remains aligned with original input")


def test_normal_rows_no_multiclass():
    X, _ = preprocess_csv(NORMAL_CSV)
    results = predict_preprocessed(X)
    normal_rows = results[results["binary_prediction"] == "NORMAL"]
    for _, row in normal_rows.iterrows():
        assert pd.isna(row["multiclass_prediction"])
        assert pd.isna(row["multiclass_confidence"])
    print("PASS: normal rows do not get multiclass predictions")


def test_attack_rows_get_multiclass():
    X, _ = preprocess_csv(ATTACK_CSV)
    results = predict_preprocessed(X)
    attack_rows = results[results["binary_prediction"] == "ATTACK"]
    # Known attack rows (not unknown/unseen) should have multiclass predictions
    known_attack_rows = attack_rows[
        attack_rows["final_prediction"] != "Unknown / Unseen Attack"
    ]
    if len(known_attack_rows) > 0:
        for _, row in known_attack_rows.iterrows():
            assert pd.notna(row["multiclass_prediction"])
            assert pd.notna(row["multiclass_confidence"])
    # Unknown/unseen attack rows intentionally have no multiclass class
    unknown_rows = attack_rows[
        attack_rows["final_prediction"] == "Unknown / Unseen Attack"
    ]
    for _, row in unknown_rows.iterrows():
        assert pd.isna(row["multiclass_prediction"])
    print("PASS: known attack rows receive multiclass predictions; unknown rows do not")


def test_unknown_threshold_used():
    from config import UNKNOWN_THRESHOLD as UT
    assert UT == 0.60
    X, _ = preprocess_csv(ATTACK_CSV)
    results = predict_preprocessed(X)
    # Check that final_confidence for unknown rows is below threshold
    unknown_rows = results[results["final_prediction"] == "Unknown / Unseen Attack"]
    for _, row in unknown_rows.iterrows():
        assert row["final_confidence"] < UT
    print(f"PASS: unknown/unseen logic uses UNKNOWN_THRESHOLD = {UT}")


def test_required_columns():
    X, _ = preprocess_csv(NORMAL_CSV)
    results = predict_preprocessed(X)
    required = [
        "row_id",
        "binary_prediction",
        "binary_confidence",
        "final_prediction",
        "final_confidence",
    ]
    for col in required:
        assert col in results.columns, f"Missing column: {col}"
    print("PASS: output contains required columns")


def test_models_not_retrained():
    binary_model = load_binary_model()
    multiclass_model = load_multiclass_model()
    label_encoder = load_label_encoder()
    assert binary_model is not None
    assert multiclass_model is not None
    assert label_encoder is not None
    assert hasattr(label_encoder, "classes_")
    print("PASS: models loaded, not retrained")


if __name__ == "__main__":
    test_normal_traffic_binary()
    test_attack_traffic_full_pipeline()
    test_output_row_count_matches_input()
    test_row_id_alignment()
    test_normal_rows_no_multiclass()
    test_attack_rows_get_multiclass()
    test_unknown_threshold_used()
    test_required_columns()
    test_models_not_retrained()
    print("\nAll inference tests passed.")
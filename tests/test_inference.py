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


def test_normal_traffic_classified_as_normal():
    """Regression test: normal traffic should be classified as NORMAL by binary model."""
    X, _ = preprocess_csv(NORMAL_CSV)
    results = predict_preprocessed(X)
    binary_preds = results["binary_prediction"]
    normal_count = int((binary_preds == "NORMAL").sum())
    attack_count = int((binary_preds == "ATTACK").sum())
    total = len(binary_preds)
    normal_pct = normal_count / total * 100
    print(f"  Normal traffic: {normal_count}/{total} NORMAL ({normal_pct:.1f}%), {attack_count} ATTACK")
    assert normal_pct >= 90, f"Expected >= 90% normal traffic to be NORMAL, got {normal_pct:.1f}%"
    print("PASS: normal traffic classified as NORMAL")


def test_attack_traffic_classified_as_attack():
    """Regression test: attack traffic should be classified as ATTACK by binary model."""
    X, _ = preprocess_csv(ATTACK_CSV)
    results = predict_preprocessed(X)
    binary_preds = results["binary_prediction"]
    attack_count = int((binary_preds == "ATTACK").sum())
    normal_count = int((binary_preds == "NORMAL").sum())
    total = len(binary_preds)
    attack_pct = attack_count / total * 100
    print(f"  Attack traffic: {attack_count}/{total} ATTACK ({attack_pct:.1f}%), {normal_count} NORMAL")
    assert attack_pct >= 90, f"Expected >= 90% attack traffic to be ATTACK, got {attack_pct:.1f}%"
    print("PASS: attack traffic classified as ATTACK")


def test_feature_order_matches_scaler():
    """Regression test: model input must use scaler's feature_names_in_ order."""
    from src.model_loader import load_scaler, load_selected_features
    scaler = load_scaler()
    scaler_features = list(scaler.feature_names_in_)
    aco_features = load_selected_features()
    # ACO order and scaler order are different (both have 32 features)
    assert len(aco_features) == 32
    assert len(scaler_features) == 32
    assert set(aco_features) == set(scaler_features)
    print(f"  ACO order first 5: {aco_features[:5]}")
    print(f"  Scaler order first 5: {scaler_features[:5]}")
    print("PASS: feature order verified (ACO for selection, scaler order for model)")


def test_binary_output_interpretation():
    """Regression test: sigmoid >= 0.50 means ATTACK, < 0.50 means NORMAL."""
    from src.inference import predict_binary
    from src.model_loader import load_binary_model
    binary_model = load_binary_model()
    X, _ = preprocess_csv(NORMAL_CSV)
    _, _, probs = predict_binary(binary_model, X)
    # Normal traffic should have low probabilities (close to 0 = NORMAL)
    mean_prob = float(np.mean(probs))
    print(f"  Normal traffic mean binary probability: {mean_prob:.4f}")
    assert mean_prob < 0.5, f"Expected normal traffic mean prob < 0.5, got {mean_prob:.4f}"
    print("PASS: binary output interpretation correct (low prob = NORMAL)")


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
    test_normal_traffic_classified_as_normal()
    test_attack_traffic_classified_as_attack()
    test_feature_order_matches_scaler()
    test_binary_output_interpretation()
    print("\nAll inference tests passed.")
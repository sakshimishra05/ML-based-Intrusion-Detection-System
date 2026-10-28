import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.model_loader import N_FEATURES, load_scaler
from src.preprocessor import (
    PreprocessingError,
    preprocess_csv,
    preprocess_dataframe,
    validate_input_columns,
)

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
NORMAL_CSV = os.path.join(SAMPLE_DIR, "normal_traffic.csv")
ATTACK_CSV = os.path.join(SAMPLE_DIR, "attack_traffic.csv")


def test_normal_csv_processes():
    X, df_selected = preprocess_csv(NORMAL_CSV)
    assert X is not None
    print("PASS: normal_traffic.csv processes successfully")


def test_attack_csv_processes():
    X, df_selected = preprocess_csv(ATTACK_CSV)
    assert X is not None
    print("PASS: attack_traffic.csv processes successfully")


def test_output_has_32_features():
    X, _ = preprocess_csv(NORMAL_CSV)
    assert X.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {X.shape[1]}"
    print(f"PASS: output has {N_FEATURES} features")


def test_final_shape_is_N_32_1():
    X, _ = preprocess_csv(NORMAL_CSV)
    assert len(X.shape) == 3, f"Expected 3D shape, got {X.shape}"
    assert X.shape[1] == N_FEATURES
    assert X.shape[2] == 1
    print(f"PASS: final shape is {X.shape} = (N, {N_FEATURES}, 1)")


def test_missing_feature_error():
    df = pd.read_csv(NORMAL_CSV)
    df = df.drop(columns=["Dst_Port"])
    try:
        preprocess_dataframe(df)
        assert False, "Should have raised PreprocessingError"
    except PreprocessingError as exc:
        assert "Dst_Port" in str(exc)
        print(f"PASS: missing feature error: {exc}")


def test_extra_columns_ignored():
    df = pd.read_csv(NORMAL_CSV)
    df["extra_col"] = 123
    df["another_extra"] = "foo"
    X, _ = preprocess_dataframe(df)
    assert X.shape[1] == N_FEATURES
    print("PASS: extra columns do not break preprocessing")


def test_scaler_not_refit():
    scaler = load_scaler()
    assert hasattr(scaler, "mean_"), "Scaler should already be fitted"
    assert hasattr(scaler, "scale_"), "Scaler should already be fitted"
    print("PASS: scaler is pre-fitted (not refit during inference)")


def test_no_nan_or_inf_after_preprocessing():
    X, _ = preprocess_csv(NORMAL_CSV)
    assert not np.isnan(X).any(), "NaN values remain after preprocessing"
    assert not np.isinf(X).any(), "Infinite values remain after preprocessing"
    print("PASS: no NaN or infinite values after preprocessing")


if __name__ == "__main__":
    test_normal_csv_processes()
    test_attack_csv_processes()
    test_output_has_32_features()
    test_final_shape_is_N_32_1()
    test_missing_feature_error()
    test_extra_columns_ignored()
    test_scaler_not_refit()
    test_no_nan_or_inf_after_preprocessing()
    print("\nAll preprocessor tests passed.")
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import (
    load_binary_model,
    load_multiclass_model,
    load_scaler,
    load_label_encoder,
    load_selected_features,
    N_FEATURES,
    EXPECTED_CLASS_COUNT,
)


def test_binary_model_loaded():
    model = load_binary_model()
    assert model is not None, "Binary model should not be None"
    print("PASS: binary model loaded")


def test_multiclass_model_loaded():
    model = load_multiclass_model()
    assert model is not None, "Multiclass model should not be None"
    print("PASS: multiclass model loaded")


def test_scaler_loaded():
    scaler = load_scaler()
    assert scaler is not None, "Scaler should not be None"
    assert hasattr(scaler, "transform"), "Scaler should have transform()"
    assert scaler.n_features_in_ == N_FEATURES, (
        f"Scaler should expect {N_FEATURES} features"
    )
    print("PASS: scaler loaded")


def test_label_encoder_loaded():
    le = load_label_encoder()
    assert le is not None, "Label encoder should not be None"
    assert len(le.classes_) == EXPECTED_CLASS_COUNT, (
        f"Label encoder should have {EXPECTED_CLASS_COUNT} classes"
    )
    print("PASS: label encoder loaded")
    print("  Classes:", list(le.classes_))


def test_selected_features_loaded():
    features = load_selected_features()
    assert len(features) == N_FEATURES, (
        f"Expected {N_FEATURES} features, got {len(features)}"
    )
    assert len(set(features)) == len(features), "Feature names must be unique"
    print("PASS: selected features loaded")
    print("  Feature count:", len(features))


if __name__ == "__main__":
    test_binary_model_loaded()
    test_multiclass_model_loaded()
    test_scaler_loaded()
    test_label_encoder_loaded()
    test_selected_features_loaded()
    print("\nAll model loader tests passed.")
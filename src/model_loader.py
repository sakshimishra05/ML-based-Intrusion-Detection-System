import os
import warnings

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

from config import (
    BINARY_MODEL_PATH,
    MULTICLASS_MODEL_PATH,
    FEATURES_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    N_FEATURES,
)

EXPECTED_CLASS_COUNT = 9


def _warn_if_missing(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required artifact not found: {path}\n"
            f"Place the file at the path above and restart the application."
        )


@st.cache_resource
def load_binary_model():
    """Load the trained binary CNN-LSTM model (normal vs attack)."""
    _warn_if_missing(BINARY_MODEL_PATH)
    try:
        model = keras.models.load_model(str(BINARY_MODEL_PATH))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load binary model from {BINARY_MODEL_PATH}: {exc}"
        ) from exc
    return model


@st.cache_resource
def load_multiclass_model():
    """Load the trained multiclass CNN-LSTM model (known attack classes)."""
    _warn_if_missing(MULTICLASS_MODEL_PATH)
    try:
        model = keras.models.load_model(str(MULTICLASS_MODEL_PATH))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load multiclass model from {MULTICLASS_MODEL_PATH}: {exc}"
        ) from exc
    return model


@st.cache_resource
def load_scaler():
    """Load the saved scaler. Only transform() should be used at inference."""
    _warn_if_missing(SCALER_PATH)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = joblib.load(str(SCALER_PATH))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load scaler from {SCALER_PATH}: {exc}"
        ) from exc

    if not hasattr(scaler, "transform"):
        raise ValueError(
            f"Scaler at {SCALER_PATH} does not have a transform() method."
        )
    if not hasattr(scaler, "n_features_in_"):
        raise ValueError(
            f"Scaler at {SCALER_PATH} is missing n_features_in_ attribute."
        )
    if scaler.n_features_in_ != N_FEATURES:
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features, "
            f"but the pipeline requires {N_FEATURES}."
        )
    return scaler


@st.cache_resource
def load_label_encoder():
    """Load the saved label encoder. Only used for decoding predictions."""
    _warn_if_missing(LABEL_ENCODER_PATH)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label_encoder = joblib.load(str(LABEL_ENCODER_PATH))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load label encoder from {LABEL_ENCODER_PATH}: {exc}"
        ) from exc

    if not hasattr(label_encoder, "classes_"):
        raise ValueError(
            f"Label encoder at {LABEL_ENCODER_PATH} is missing classes_ attribute."
        )
    if len(label_encoder.classes_) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Label encoder expects {EXPECTED_CLASS_COUNT} classes, "
            f"but found {len(label_encoder.classes_)}."
        )
    return label_encoder


@st.cache_resource
def load_selected_features():
    """Load the ACO-selected 32 feature names in their saved order."""
    _warn_if_missing(FEATURES_PATH)
    try:
        features_df = pd.read_csv(str(FEATURES_PATH))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read selected features from {FEATURES_PATH}: {exc}"
        ) from exc

    if "Feature" not in features_df.columns:
        raise ValueError(
            f"Selected features file {FEATURES_PATH} is missing the 'Feature' column."
        )

    features = features_df["Feature"].tolist()

    if len(features) != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} selected features, but found {len(features)}."
        )

    if len(set(features)) != len(features):
        raise ValueError(
            f"Selected features in {FEATURES_PATH} contain duplicate names."
        )

    return features


def load_all_artifacts():
    """Load and validate all models and preprocessing artifacts."""
    return {
        "binary_model": load_binary_model(),
        "multiclass_model": load_multiclass_model(),
        "scaler": load_scaler(),
        "label_encoder": load_label_encoder(),
        "selected_features": load_selected_features(),
    }
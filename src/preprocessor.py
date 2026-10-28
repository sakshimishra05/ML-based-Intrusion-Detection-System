import warnings

import numpy as np
import pandas as pd

from config import N_FEATURES
from src.model_loader import load_scaler, load_selected_features


class PreprocessingError(Exception):
    """Raised when input CSV cannot be preprocessed for inference."""


def validate_input_columns(df, required_features):
    """Check that df contains all required feature columns."""
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise PreprocessingError(
            "Missing required features: " + ", ".join(missing)
        )
    return True


def _to_numeric_and_report(df, features):
    """Convert selected columns to numeric and report non-numeric rows."""
    cleaned = df[features].copy()

    for col in features:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    nan_mask = cleaned.isna()
    if nan_mask.any().any():
        bad_rows = []
        for idx in cleaned.index[nan_mask.any(axis=1)]:
            bad_cols = [c for c in features if nan_mask.loc[idx, c]]
            bad_rows.append(f"row {idx}: {', '.join(bad_cols)}")
        raise PreprocessingError(
            "Non-numeric or missing values found in: " + "; ".join(bad_rows[:10])
        )

    inf_mask = np.isinf(cleaned.values)
    if inf_mask.any():
        bad_rows = []
        for row_idx in range(inf_mask.shape[0]):
            if inf_mask[row_idx].any():
                bad_cols = [features[j] for j in range(inf_mask.shape[1]) if inf_mask[row_idx, j]]
                bad_rows.append(f"row {row_idx}: {', '.join(bad_cols)}")
        raise PreprocessingError(
            "Infinite values found in: " + "; ".join(bad_rows[:10])
        )

    return cleaned


def preprocess_dataframe(df):
    """Validate and preprocess a DataFrame into CNN-LSTM input shape.

    Returns:
        X: numpy array of shape (N, 32, 1) ready for model inference
        df_selected: DataFrame with the 32 features in ACO order (pre-scale)
    """
    if df is None or len(df) == 0:
        raise PreprocessingError(
            "The uploaded CSV is empty. Please upload a file with data rows."
        )

    selected_features = load_selected_features()
    validate_input_columns(df, selected_features)

    df_selected = _to_numeric_and_report(df, selected_features)

    # Apply the saved scaler with transform() only -- never fit.
    # The scaler requires columns in its feature_names_in_ order.
    scaler = load_scaler()
    scaler_features = list(getattr(scaler, "feature_names_in_", selected_features))

    df_for_scaler = df_selected[scaler_features]
    scaled = scaler.transform(df_for_scaler)

    # scaler.transform() returns columns in scaler.feature_names_in_ order.
    # Reorder back to the saved ACO order expected by the model.
    if scaler_features != selected_features:
        scaled_df = pd.DataFrame(scaled, columns=scaler_features)
        scaled = scaled_df[selected_features].values

    # Final check: no NaN or infinite values after scaling
    if np.isnan(scaled).any() or np.isinf(scaled).any():
        raise PreprocessingError(
            "NaN or infinite values remain after scaling."
        )

    # Reshape for CNN-LSTM input: (number_of_rows, 32, 1)
    X = scaled.reshape((scaled.shape[0], N_FEATURES, 1))

    return X, df_selected


def preprocess_csv(file_or_path):
    """Read a CSV file and preprocess it for model inference."""
    df = pd.read_csv(file_or_path)
    return preprocess_dataframe(df)
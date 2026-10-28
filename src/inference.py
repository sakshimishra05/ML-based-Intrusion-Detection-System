import math

import numpy as np
import pandas as pd

from config import (
    BINARY_THRESHOLD,
    PREDICTION_BATCH_SIZE,
    UNKNOWN_THRESHOLD,
)
from src.model_loader import (
    load_binary_model,
    load_label_encoder,
    load_multiclass_model,
)
from src.preprocessor import preprocess_dataframe


class InferenceError(Exception):
    """Raised when inference cannot be completed."""


def _predict_in_batches(model, X, batch_size):
    """Run model.predict in batches to avoid memory issues."""
    n = X.shape[0]
    outputs = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = X[start:end]
        out = model.predict(batch, verbose=0)
        outputs.append(out)
    if len(outputs) == 1:
        return outputs[0]
    return np.vstack(outputs)


def predict_binary(binary_model, X, threshold=BINARY_THRESHOLD):
    """Run binary CNN-LSTM prediction.

    Returns:
        predictions: array of "NORMAL" / "ATTACK"
        confidences: array of probabilities
        probabilities: raw model output probabilities
    """
    probabilities = _predict_in_batches(binary_model, X, PREDICTION_BATCH_SIZE).flatten()
    predictions = np.where(probabilities >= threshold, "ATTACK", "NORMAL")
    confidences = np.where(
        predictions == "ATTACK", probabilities, 1.0 - probabilities
    )
    return predictions, confidences, probabilities


def predict_multiclass(multiclass_model, label_encoder, X):
    """Run multiclass CNN-LSTM prediction on attack rows.

    Returns:
        class_names: array of predicted class names
        confidences: array of max class probabilities
    """
    probs = _predict_in_batches(multiclass_model, X, PREDICTION_BATCH_SIZE)
    class_indices = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(class_indices)), class_indices]
    class_names = label_encoder.inverse_transform(class_indices)
    return class_names, confidences


def predict_preprocessed(X):
    """Run the full IDS pipeline on preprocessed input.

    Args:
        X: numpy array of shape (N, 32, 1)

    Returns:
        pandas DataFrame with detection results
    """
    binary_model = load_binary_model()
    multiclass_model = load_multiclass_model()
    label_encoder = load_label_encoder()

    n = X.shape[0]

    # Step 1: binary prediction on all rows
    binary_preds, binary_confs, _ = predict_binary(binary_model, X)

    # Prepare output
    final_preds = []
    final_confs = []
    multiclass_preds = []
    multiclass_confs = []

    # Identify attack rows for multiclass
    attack_mask = binary_preds == "ATTACK"
    attack_indices = np.where(attack_mask)[0]

    # Default: all normal
    for i in range(n):
        final_preds.append("Normal")
        final_confs.append(float(binary_confs[i]))
        multiclass_preds.append(None)
        multiclass_confs.append(None)

    # Step 2: multiclass prediction on attack rows only
    if len(attack_indices) > 0:
        X_attacks = X[attack_indices]
        mc_class_names, mc_confs = predict_multiclass(
            multiclass_model, label_encoder, X_attacks
        )

        for pos, row_idx in enumerate(attack_indices):
            mc_conf = float(mc_confs[pos])
            mc_class = mc_class_names[pos]

            # Step 3: unknown/unseen detection
            if mc_conf < UNKNOWN_THRESHOLD:
                final_preds[row_idx] = "Unknown / Unseen Attack"
                final_confs[row_idx] = mc_conf
                multiclass_preds[row_idx] = None
                multiclass_confs[row_idx] = None
            else:
                final_preds[row_idx] = mc_class
                final_confs[row_idx] = mc_conf
                multiclass_preds[row_idx] = mc_class
                multiclass_confs[row_idx] = mc_conf

    results = pd.DataFrame(
        {
            "row_id": list(range(n)),
            "binary_prediction": binary_preds.tolist(),
            "binary_confidence": [float(c) for c in binary_confs],
            "final_prediction": final_preds,
            "final_confidence": final_confs,
            "multiclass_prediction": multiclass_preds,
            "multiclass_confidence": multiclass_confs,
        }
    )

    return results


def run_detection(df_or_csv):
    """Run detection from a DataFrame or CSV file path.

    Args:
        df_or_csv: pandas DataFrame or path to CSV file

    Returns:
        results: pandas DataFrame with detection results
        X: preprocessed input array (N, 32, 1)
        df_selected: DataFrame with selected features in ACO order
    """
    if isinstance(df_or_csv, str):
        df = pd.read_csv(df_or_csv)
    else:
        df = df_or_csv
    X, df_selected = preprocess_dataframe(df)
    results = predict_preprocessed(X)
    return results, X, df_selected
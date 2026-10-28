# ML-Based Intrusion Detection System

Local inference and demonstration application for the machine learning-based
intrusion detection system trained on the IoTID20 dataset.

## Project overview

This application loads already-trained deep learning models and provides a
Streamlit UI for uploading network traffic CSV files and displaying
intrusion-detection results. No model training is performed here.

## Repository structure

app.py
config.py
requirements.txt
README.md
src/__init__.py
src/model_loader.py
src/preprocessor.py
src/inference.py
tests/
models/
multiclass_models/
preprocessing_artifacts/
sample_data/

## Required files (place locally, not committed to GitHub)

The trained models and preprocessing artifacts are not committed to GitHub.
Place them in the project folder before running the application:

- `models/CNN_LSTM_IoTID20.keras` — binary CNN-LSTM model (normal vs attack)
- `multiclass_models/CNN_LSTM_IoTID20_multiclass_aco32.keras` — multiclass CNN-LSTM model (known attack classes)
- `preprocessing_artifacts/aco_selected_features.csv` — ACO-selected 32 features
- `preprocessing_artifacts/aco_standard_scaler_SMOTE.pkl` — saved scaler
- `preprocessing_artifacts/multiclass_label_encoder.pkl` — saved label encoder

Sample CSV files are provided under `sample_data/`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. Upload a network traffic CSV (or choose a sample file).
2. The app validates that the required ACO-32 feature columns are present.
3. The saved scaler is loaded and applied with `transform()` only — it is never refit.
4. Data is reshaped to `(N, 32, 1)` for the deep learning models.
5. The binary CNN-LSTM model predicts normal vs attack.
6. If the flow is classified as an attack, the multiclass CNN-LSTM model
   predicts the known attack class and confidence.
7. If the highest multiclass probability is below the configured threshold,
   the flow is reported as UNKNOWN / potential unseen attack.

## Input CSV requirements

The uploaded CSV must contain the 32 ACO-selected features used during
training. Extra columns are ignored. Missing required columns produce a
clear error message listing the missing columns.

## Unknown / unseen attacks

The multiclass model only knows the IoTID20 classes. This prototype uses a
confidence-based threshold: if the highest softmax probability is below the
threshold, the flow is reported as UNKNOWN / potential unseen attack. This is
a heuristic for demonstration purposes and is not a scientifically proven
zero-day detection method.

## System Architecture

```
CSV traffic
  → 32-feature validation (ACO-selected features)
  → saved scaler transform() only (never refit)
  → reshape (N, 32, 1)
  → Binary CNN-LSTM: Normal / Attack
      → if Normal: result = Normal
      → if Attack: Multiclass CNN-LSTM
          → known attack class + confidence
          → if confidence < threshold: Unknown / Unseen Attack
```

The binary CNN-LSTM model (`models/CNN_LSTM_IoTID20.keras`) runs first and
decides normal vs attack. Only traffic classified as attack is passed to the
multiclass CNN-LSTM model
(`multiclass_models/CNN_LSTM_IoTID20_multiclass_aco32.keras`) for known
attack classification. Normal traffic is never sent to the multiclass model.

The unknown/unseen result is a **confidence-based anomaly indicator**, not a
mathematically proven zero-day detection. It means the multiclass model's
highest softmax probability fell below the configured threshold.

## Demo

1. Start the application:

   ```bash
   streamlit run app.py
   ```

2. Upload a CSV containing the required 32 network-flow features, or select
   one of the sample traffic buttons (Normal Traffic / Attack Traffic).
3. Click **Analyze Traffic**.
4. Review the detection summary (total records, normal, known attacks,
   unknown/unseen attacks).
5. Inspect per-row predictions in the results table.
6. Download the prediction results as CSV.

## Limitations

- No model training or retraining is performed.
- SMOTE and ACO are not run during inference.
- The scaler is never refit during inference.
- Unknown detection is confidence-based only.

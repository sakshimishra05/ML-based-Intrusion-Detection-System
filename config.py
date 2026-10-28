from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
MULTICLASS_MODEL_DIR = BASE_DIR / "multiclass_models"
ARTIFACT_DIR = BASE_DIR / "preprocessing_artifacts"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

BINARY_MODEL_PATH = MODEL_DIR / "CNN_LSTM_IoTID20.keras"
MULTICLASS_MODEL_PATH = MULTICLASS_MODEL_DIR / "CNN_LSTM_IoTID20_multiclass_aco32.keras"

FEATURES_PATH = ARTIFACT_DIR / "aco_selected_features.csv"
SCALER_PATH = ARTIFACT_DIR / "aco_standard_scaler_SMOTE.pkl"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "multiclass_label_encoder.pkl"

N_FEATURES = 32
SEQUENCE_SHAPE = (N_FEATURES, 1)
UNKNOWN_THRESHOLD = 0.40
BINARY_THRESHOLD = 0.50
BATCH_ATTACK_THRESHOLD = 0.05
PREDICTION_BATCH_SIZE = 1024
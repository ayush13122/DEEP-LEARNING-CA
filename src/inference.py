"""Small artifact-loading helpers used by the dashboard and tests."""
from pathlib import Path
import joblib
import tensorflow as tf


def load_artifacts(root: str | Path):
    root = Path(root)
    artifacts = root / "artifacts"
    return {
        "lstm": tf.keras.models.load_model(artifacts / "lstm_model.keras"),
        "cnn": tf.keras.models.load_model(artifacts / "cnn_model.keras"),
        "feature_scaler": joblib.load(artifacts / "feature_scaler.joblib"),
        "target_scaler": joblib.load(artifacts / "target_scaler.joblib"),
    }

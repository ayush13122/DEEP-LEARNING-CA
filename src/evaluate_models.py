"""Read-only validation of generated evaluation artifacts."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    pred = pd.read_csv(ROOT / "outputs" / "test_predictions.csv", parse_dates=["target_timestamp"])
    metrics = json.loads((ROOT / "outputs" / "model_metrics.json").read_text())
    assert pred.target_timestamp.is_monotonic_increasing and pred.target_timestamp.is_unique
    for label, col in [("LSTM", "lstm_prediction"), ("1D-CNN", "cnn_prediction"), ("Persistence", "persistence_prediction")]:
        actual = pred.actual_ac_power.to_numpy(); forecast = pred[col].to_numpy()
        observed = {"MAE": mean_absolute_error(actual, forecast), "RMSE": mean_squared_error(actual, forecast) ** .5, "R2": r2_score(actual, forecast)}
        assert all(np.isclose(observed[k], metrics[label][k], atol=1e-5) for k in observed), label
    print(f"Evaluation artifacts verified: {len(pred)} aligned samples.")

if __name__ == "__main__": main()

"""Train and evaluate LSTM and 1D-CNN models for Model 2 only.

Run with: python -m src.train_models
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import (LOOKBACK, build_sequences, chronological_masks, feature_columns,
                               load_and_audit, split_metadata, write_json)
from src.model_cnn import build_cnn
from src.model_lstm import build_lstm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Model2_Intraday_NextHour.csv"
ARTIFACTS, OUTPUTS, FIGURES = ROOT / "artifacts", ROOT / "outputs", ROOT / "outputs" / "figures"


def set_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"MAE": float(mean_absolute_error(y, pred)), "RMSE": float(mean_squared_error(y, pred) ** 0.5), "R2": float(r2_score(y, pred))}


def summarise_model(model: tf.keras.Model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def plot_results(predictions: pd.DataFrame, histories: dict[str, pd.DataFrame], metric_df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    # plot an interpretable, bounded test slice while retaining all samples in the CSV
    preview = predictions.iloc[: min(600, len(predictions))]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(preview["target_timestamp"], preview["actual_ac_power"], label="Actual", color="#1f2937", linewidth=1.5)
    ax.plot(preview["target_timestamp"], preview["lstm_prediction"], label="LSTM", color="#f59e0b", alpha=.9)
    ax.plot(preview["target_timestamp"], preview["cnn_prediction"], label="1D-CNN", color="#2563eb", alpha=.9)
    ax.set(title="Actual and predicted AC power (first 600 test samples)", ylabel="AC power (kW)", xlabel="Target timestamp")
    ax.legend(ncol=3); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIGURES / "actual_vs_predicted.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    for col, color in [("lstm_prediction", "#f59e0b"), ("cnn_prediction", "#2563eb"), ("persistence_prediction", "#64748b")]:
        ax.hist(predictions[col] - predictions["actual_ac_power"], bins=60, alpha=.46, label=col.replace("_prediction", "").upper(), color=color)
    ax.axvline(0, color="black", linewidth=.8); ax.set(title="Test residual distribution", xlabel="Prediction minus actual (kW)", ylabel="Samples"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "residual_distribution.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(metric_df["model"], metric_df["RMSE"], color=["#f59e0b", "#2563eb", "#64748b"])
    ax.set(title="Final test RMSE comparison", ylabel="RMSE (kW)"); fig.tight_layout(); fig.savefig(FIGURES / "metric_comparison.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, history) in zip(axes, histories.items()):
        ax.plot(history["loss"], label="Training loss", color="#2563eb")
        ax.plot(history["val_loss"], label="Validation loss", color="#f59e0b")
        ax.set(title=f"{name} loss", xlabel="Epoch", ylabel="MSE"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "training_validation_loss.png", dpi=160); plt.close(fig)


def main() -> None:
    set_seeds()
    ARTIFACTS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    df, audit = load_and_audit(DATA)
    features = feature_columns(df)
    X, y, feature_ends, target_times, persistence, sequence_stats = build_sequences(df, features, LOOKBACK)
    audit.update(sequence_stats)
    masks = chronological_masks(target_times)
    audit["split_definition"] = split_metadata(target_times, masks)
    write_json(audit, OUTPUTS / "data_quality_audit.json")

    # Fit scalers strictly on training tensor rows / labels. Transforming test does not refit.
    f_scaler, y_scaler = StandardScaler(), StandardScaler()
    f_scaler.fit(X[masks["train"]].reshape(-1, X.shape[-1]))
    y_scaler.fit(y[masks["train"]].reshape(-1, 1))
    X_scaled = f_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype("float32")
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).astype("float32")
    joblib.dump(f_scaler, ARTIFACTS / "feature_scaler.joblib"); joblib.dump(y_scaler, ARTIFACTS / "target_scaler.joblib")
    train, val, test = masks["train"], masks["validation"], masks["test"]
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=.5, min_lr=1e-5)]
    train_args = dict(x=X_scaled[train], y=y_scaled[train], validation_data=(X_scaled[val], y_scaled[val]), epochs=40, batch_size=128, verbose=2, callbacks=callbacks)
    lstm = build_lstm((LOOKBACK, len(features))); history_lstm = lstm.fit(**train_args)
    cnn = build_cnn((LOOKBACK, len(features))); history_cnn = cnn.fit(**train_args)
    lstm.save(ARTIFACTS / "lstm_model.keras"); cnn.save(ARTIFACTS / "cnn_model.keras")
    histories = {"LSTM": pd.DataFrame(history_lstm.history), "1D-CNN": pd.DataFrame(history_cnn.history)}
    histories["LSTM"].to_csv(OUTPUTS / "lstm_history.csv", index=False); histories["1D-CNN"].to_csv(OUTPUTS / "cnn_history.csv", index=False)
    pred_lstm = y_scaler.inverse_transform(lstm.predict(X_scaled[test], verbose=0)).ravel()
    pred_cnn = y_scaler.inverse_transform(cnn.predict(X_scaled[test], verbose=0)).ravel()
    actual, base = y[test], persistence[test]
    test_times = target_times[test]
    if not np.array_equal(test_times.values, target_times[test].values): raise RuntimeError("Test timestamps disagree")
    prediction_df = pd.DataFrame({"target_timestamp": test_times, "actual_ac_power": actual, "lstm_prediction": pred_lstm, "cnn_prediction": pred_cnn, "persistence_prediction": base}).sort_values("target_timestamp")
    prediction_df.to_csv(OUTPUTS / "test_predictions.csv", index=False)
    rows = []
    for name, pred in [("LSTM", pred_lstm), ("1D-CNN", pred_cnn), ("Persistence", base)]: rows.append({"model": name, **metrics(actual, pred)})
    metric_df = pd.DataFrame(rows)
    metric_df.to_csv(OUTPUTS / "model_metrics.csv", index=False)
    payload = {r["model"]: {k: round(v, 6) for k, v in r.items() if k != "model"} for r in rows}
    payload["test_sample_count"] = int(len(actual)); payload["units"] = {"MAE": "kW", "RMSE": "kW", "R2": "unitless"}
    write_json(payload, OUTPUTS / "model_metrics.json")
    config = {"project": "Model 2 intraday next-hour solar forecasting", "lookback": LOOKBACK, "input_shape": [LOOKBACK, len(features)], "features": features, "target": "target_ac_power", "seed": 42, "epochs_requested": 40, "batch_size": 128, "split": split_metadata(target_times, masks), "scaler_fit_scope": "training sequences and labels only", "architectures": {"lstm": summarise_model(lstm), "cnn": summarise_model(cnn)}}
    write_json(config, ARTIFACTS / "model_config.json")
    (ARTIFACTS / "lstm_architecture.txt").write_text(config["architectures"]["lstm"], encoding="utf-8")
    (ARTIFACTS / "cnn_architecture.txt").write_text(config["architectures"]["cnn"], encoding="utf-8")
    plot_results(prediction_df, histories, metric_df)
    print(metric_df.to_string(index=False)); print(f"Saved {len(prediction_df)} aligned final-test predictions.")


if __name__ == "__main__": main()

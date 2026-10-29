"""Validated, leakage-aware preparation for Model 2 next-hour sequences."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOOKBACK = 24
TARGET = "target_ac_power"
TIMESTAMP = "timestamp"
TARGET_TIMESTAMP = "target_timestamp"


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numerical modelling features, explicitly excluding labels and timestamps."""
    excluded = {TIMESTAMP, TARGET_TIMESTAMP, TARGET}
    cols = [c for c in frame.columns if c not in excluded]
    if not cols or TARGET in cols or TARGET_TIMESTAMP in cols:
        raise ValueError("Feature exclusion failed")
    return cols


def load_and_audit(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read data, enforce timestamp/target integrity and return a serialisable audit."""
    path = Path(path)
    df = pd.read_csv(path, parse_dates=[TIMESTAMP, TARGET_TIMESTAMP])
    required = {TIMESTAMP, TARGET_TIMESTAMP, TARGET}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    before = len(df)
    out_of_order = not df[TIMESTAMP].is_monotonic_increasing
    df = df.sort_values(TIMESTAMP, kind="stable").reset_index(drop=True)
    duplicate_count = int(df[TIMESTAMP].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Duplicate feature timestamps detected: {duplicate_count}")
    null_count = int(df.isna().sum().sum())
    if null_count:
        raise ValueError(f"Missing values detected: {null_count}")
    alignment = df[TARGET_TIMESTAMP].eq(df[TIMESTAMP] + pd.Timedelta(hours=1))
    invalid_targets = int((~alignment).sum())
    if invalid_targets:
        raise ValueError(f"Invalid one-hour target alignment in {invalid_targets} rows")
    steps = df[TIMESTAMP].diff()
    gap_count = int(((steps.notna()) & (steps != pd.Timedelta(hours=1))).sum())
    audit: dict[str, Any] = {
        "source_file": str(path.name), "rows_loaded": before, "rows_after_sort": len(df),
        "column_count": int(df.shape[1]), "feature_count": len(feature_columns(df)),
        "feature_timestamp_start": str(df[TIMESTAMP].min()), "feature_timestamp_end": str(df[TIMESTAMP].max()),
        "target_timestamp_start": str(df[TARGET_TIMESTAMP].min()), "target_timestamp_end": str(df[TARGET_TIMESTAMP].max()),
        "null_cells": null_count, "duplicate_timestamps": duplicate_count,
        "input_was_out_of_order": out_of_order, "irregular_timestamp_steps": gap_count,
        "invalid_target_alignment_rows": invalid_targets, "lookback_hours": LOOKBACK,
        "target_definition": "target_ac_power at target_timestamp, exactly one hour after the final input timestamp",
    }
    return df, audit


def build_sequences(df: pd.DataFrame, features: list[str], lookback: int = LOOKBACK) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray, dict[str, int]]:
    """Build only genuinely consecutive windows; no sequence crosses a missing hour."""
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    values = df[features].to_numpy(dtype=np.float32)
    labels = df[TARGET].to_numpy(dtype=np.float32)
    times = pd.DatetimeIndex(df[TIMESTAMP])
    target_times = pd.DatetimeIndex(df[TARGET_TIMESTAMP])
    xs, ys, feature_ends, label_times, persistence = [], [], [], [], []
    rejected_gaps = 0
    candidates = max(0, len(df) - lookback + 1)
    one_hour = pd.Timedelta(hours=1)
    for end in range(lookback - 1, len(df)):
        start = end - lookback + 1
        window_times = times[start:end + 1]
        if not bool(np.all(np.diff(window_times.asi8) == one_hour.value)):
            rejected_gaps += 1
            continue
        xs.append(values[start:end + 1])
        ys.append(labels[end])
        feature_ends.append(times[end])
        label_times.append(target_times[end])
        persistence.append(values[end, features.index("ac_power__5069")])
    if not xs:
        raise ValueError("No valid sequences were created")
    stats = {"potential_windows": candidates, "valid_sequences": len(xs), "rejected_for_timestamp_gaps": rejected_gaps}
    return (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
            pd.DatetimeIndex(feature_ends), pd.DatetimeIndex(label_times), np.asarray(persistence, dtype=np.float32), stats)


def chronological_masks(target_times: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Use original Model 2 target-timestamp boundaries."""
    target_times = pd.DatetimeIndex(target_times)
    train_end = pd.Timestamp("2015-12-31 23:59:59")
    val_start, val_end = pd.Timestamp("2016-01-01"), pd.Timestamp("2016-12-31 23:59:59")
    test_start = pd.Timestamp("2017-01-01")
    masks = {
        "train": np.asarray(target_times <= train_end),
        "validation": np.asarray((target_times >= val_start) & (target_times <= val_end)),
        "test": np.asarray(target_times >= test_start),
    }
    if sum(int(v.sum()) for v in masks.values()) != len(target_times):
        raise ValueError("A sequence falls outside the declared chronological partitions")
    return masks


def split_metadata(target_times: pd.DatetimeIndex, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "assignment_basis": "target_timestamp", "training_end": "2015-12-31 23:59:59",
        "validation_start": "2016-01-01", "validation_end": "2016-12-31 23:59:59", "test_start": "2017-01-01",
        "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "actual_ranges": {name: {"start": str(target_times[mask].min()), "end": str(target_times[mask].max())} for name, mask in masks.items()},
    }


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

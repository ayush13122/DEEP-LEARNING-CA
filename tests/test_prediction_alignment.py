from pathlib import Path
import json
import pandas as pd
import pytest

@pytest.mark.skipif(not Path('outputs/test_predictions.csv').exists(), reason='run training first')
def test_saved_predictions_are_sorted_and_aligned():
    p=pd.read_csv('outputs/test_predictions.csv', parse_dates=['target_timestamp'])
    assert p.target_timestamp.is_monotonic_increasing and p.target_timestamp.is_unique
    assert list(p.columns) == ['target_timestamp','actual_ac_power','lstm_prediction','cnn_prediction','persistence_prediction']
    assert len(p) == json.loads(Path('outputs/model_metrics.json').read_text())['test_sample_count']

@pytest.mark.skipif(not Path('artifacts/lstm_model.keras').exists(), reason='run training first')
def test_saved_model_artifacts_exist():
    assert Path('artifacts/cnn_model.keras').exists()
    assert Path('artifacts/feature_scaler.joblib').exists()
    assert Path('artifacts/target_scaler.joblib').exists()

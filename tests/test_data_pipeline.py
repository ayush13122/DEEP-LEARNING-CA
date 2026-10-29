import numpy as np
import pandas as pd
from src.data_pipeline import build_sequences, chronological_masks, feature_columns, load_and_audit

def test_dataset_schema_alignment_and_feature_exclusion():
    df, audit = load_and_audit('data/Model2_Intraday_NextHour.csv')
    features = feature_columns(df)
    assert df.shape == (56275, 30)
    assert audit['invalid_target_alignment_rows'] == 0
    assert 'target_ac_power' not in features and 'target_timestamp' not in features

def test_gap_windows_are_rejected_and_shape_is_24_steps():
    times = pd.to_datetime(['2020-01-01 00:00','2020-01-01 01:00','2020-01-01 02:00','2020-01-01 04:00','2020-01-01 05:00'])
    df = pd.DataFrame({'timestamp':times,'ac_power__5069':range(5),'weather':range(5),'target_timestamp':times+pd.Timedelta(hours=1),'target_ac_power':range(5)})
    X,y,ends,targets,persist,stats = build_sequences(df, feature_columns(df), lookback=3)
    assert X.shape == (1,3,2) and stats['rejected_for_timestamp_gaps'] == 2
    assert targets[0] == ends[0] + pd.Timedelta(hours=1)

def test_chronological_masks_cover_every_sample():
    ts = pd.DatetimeIndex(['2015-12-31 23:00','2016-05-01','2017-01-01'])
    masks=chronological_masks(ts)
    assert np.array_equal(masks['train'], [True,False,False])
    assert sum(x.sum() for x in masks.values()) == 3

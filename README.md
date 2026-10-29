# Comparative Analysis of LSTM and 1D CNN for Model 2 Intraday Next Hour Solar Power Forecasting

This independent Deep Learning CA1 project evaluates two TensorFlow/Keras regressors for Model 2 intraday solar-power forecasting. It is limited to the Model 2 dataset: current plant power and contemporaneous historical environmental observations from time *t* are used to predict AC power at *t+1*.

## Reproducible workflow

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train_models
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\streamlit.exe run app.py
```

Training writes models, scalers, timestamp-aligned predictions, charts, data-quality audit and metrics under `artifacts/` and `outputs/`. The dashboard loads these saved artifacts only; it never retrains on page refresh.

## Methodology

- **Input:** 24 genuinely consecutive hourly Model 2 feature rows (27 features).
- **Label:** `target_ac_power` on the final input row, verified as exactly one hour ahead.
- **Gaps:** A window crossing a timestamp gap is discarded; no values are interpolated.
- **Split:** target timestamps through 2015 for training, 2016 for validation, and 2017 through the final available timestamp for final testing.
- **Scaling:** feature and target scalers are fit only on the training partition.
- **Models:** LSTM(64) with dense head, and temporal Conv1D(32→64) with global pooling. Persistence is a separate sanity-check baseline.

## Repository layout

- `src/`: validated preprocessing, architectures, training/evaluation and artifact loading.
- `data/`: the copied canonical Model 2 CSV.
- `artifacts/`: saved `.keras` models, scalers and model configuration after training.
- `outputs/`: metrics, aligned predictions, audit, histories and figures.
- `reports/`: Word report and concise viva preparation guide.
- `tests/`: data-pipeline and artifact-alignment tests.

This is historical retrospective evaluation, not a claim of live SCADA forecasting. The dataset ends in January 2018; weather/reanalysis variables are historical observations, not operational future weather forecasts.

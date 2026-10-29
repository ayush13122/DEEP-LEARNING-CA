"""Streamlit dashboard for completed historical Model 2 deep-learning experiment."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Solar DL CA1", page_icon="☀️", layout="wide")
st.markdown("""<style>.stApp{background:#fbfdff}.hero{padding:1.3rem 0}.stMetric{background:white;border:1px solid #e5e7eb;border-radius:10px;padding:12px}</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    metrics = pd.read_csv(ROOT / "outputs" / "model_metrics.csv")
    predictions = pd.read_csv(ROOT / "outputs" / "test_predictions.csv", parse_dates=["target_timestamp"])
    audit = json.loads((ROOT / "outputs" / "data_quality_audit.json").read_text())
    config = json.loads((ROOT / "artifacts" / "model_config.json").read_text())
    raw = pd.read_csv(ROOT / "data" / "Model2_Intraday_NextHour.csv", parse_dates=["timestamp", "target_timestamp"])
    return metrics, predictions, audit, config, raw

def main():
    required = [ROOT / "outputs" / "model_metrics.csv", ROOT / "outputs" / "test_predictions.csv", ROOT / "artifacts" / "lstm_model.keras", ROOT / "artifacts" / "cnn_model.keras"]
    if not all(p.exists() for p in required):
        st.error("Saved training artifacts are unavailable. Run `python -m src.train_models` before launching the dashboard.")
        return
    metrics, pred, audit, config, raw = load_data()
    st.sidebar.title("☀️ Solar DL CA1")
    section = st.sidebar.radio("Navigate", ["Project overview", "Dataset and methodology", "Model comparison", "Historical forecast explorer", "Model explanation", "Conclusions and limitations"])
    if section == "Project overview":
        st.markdown("<div class='hero'><h1>Intraday next-hour solar forecasting</h1><p>Comparative analysis of LSTM and 1D-CNN models using Model 2 only.</p></div>", unsafe_allow_html=True)
        st.info("Historical replay, not live telemetry. The dataset ends in January 2018.")
        st.write("Each forecast consumes 24 consecutive hourly observations ending at time t and predicts observed AC power at t+1. Current AC power is a permitted historical input; target power and target timestamp are never inputs.")
    elif section == "Dataset and methodology":
        st.header("Dataset and methodology")
        c1,c2,c3,c4=st.columns(4); c1.metric("Rows", f"{audit['rows_loaded']:,}"); c2.metric("Features", audit['feature_count']); c3.metric("Valid windows", f"{audit['valid_sequences']:,}"); c4.metric("Gap-rejected windows", f"{audit['rejected_for_timestamp_gaps']:,}")
        st.write(f"Feature timestamps: {audit['feature_timestamp_start']} to {audit['feature_timestamp_end']}. Every retained label is verified to be exactly one hour ahead.")
        st.json({k:audit[k] for k in ['duplicate_timestamps','null_cells','irregular_timestamp_steps','invalid_target_alignment_rows','split_definition']})
        st.caption(f"Model input shape: {tuple(config['input_shape'])}; splits are assigned by target timestamp.")
    elif section == "Model comparison":
        st.header("Deep learning model comparison")
        model_metrics=metrics.set_index('model')
        cards=st.columns(3)
        for card,name in zip(cards,['LSTM','1D-CNN','Persistence']):
            card.subheader(name); card.metric("MAE", f"{model_metrics.loc[name,'MAE']:.3f} kW"); card.metric("RMSE", f"{model_metrics.loc[name,'RMSE']:.3f} kW"); card.metric("R²", f"{model_metrics.loc[name,'R2']:.4f}")
        st.plotly_chart(px.line(pred.iloc[:600], x='target_timestamp', y=['actual_ac_power','lstm_prediction','cnn_prediction'], labels={'value':'AC power (kW)','variable':'Series'}, title='Actual and predictions: first 600 final-test samples'), use_container_width=True)
        st.plotly_chart(px.bar(metrics, x='model', y='RMSE', color='model', title='Final test RMSE'), use_container_width=True)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.caption("Persistence is a sanity-check baseline: next-hour prediction equals the final observed AC power in the 24-hour input window.")
    elif section == "Historical forecast explorer":
        st.header("Historical forecast explorer")
        choices = pred.target_timestamp.dt.strftime('%Y-%m-%d %H:%M').tolist()
        selected = st.selectbox("Choose final-test target timestamp", choices, index=0)
        row = pred.iloc[choices.index(selected)]
        context_end = row.target_timestamp - pd.Timedelta(hours=1)
        context = raw[(raw.timestamp >= context_end-pd.Timedelta(hours=23)) & (raw.timestamp <= context_end)][['timestamp','ac_power__5069','poa_irradiance__5061','ambient_temp__5062']]
        st.caption(f"Twenty-four hour context ending {context_end}; evaluated target {row.target_timestamp}.")
        cols=st.columns(4)
        for c,(label,key) in zip(cols,[('Actual','actual_ac_power'),('LSTM','lstm_prediction'),('1D-CNN','cnn_prediction'),('Persistence','persistence_prediction')]): c.metric(label, f"{row[key]:.2f} kW")
        st.plotly_chart(px.line(context, x='timestamp', y='ac_power__5069', title='Observed AC power context'), use_container_width=True)
        st.dataframe(context, use_container_width=True, hide_index=True)
    elif section == "Model explanation":
        st.header("How the two models learn")
        st.subheader("LSTM")
        st.write("The LSTM processes the 24 hourly records in order and uses gated memory to retain useful earlier context such as the daily generation cycle and recent ramps.")
        st.code("24 × 27 features → LSTM(64) → Dense(32, ReLU) → next-hour AC power")
        st.subheader("1D-CNN")
        st.write("The 1D-CNN slides filters along the hourly axis. It learns local patterns such as rapid irradiance changes and short generation ramps, then pools them into a forecast.")
        st.code("24 × 27 features → Conv1D(32) → Conv1D(64) → global pooling → Dense(32) → next-hour AC power")
    else:
        st.header("Conclusions and limitations")
        winner = metrics.loc[metrics.model.isin(['LSTM','1D-CNN'])].sort_values('RMSE').iloc[0]
        st.write(f"On the final 2017–January 2018 historical test period, **{winner.model}** has the lower observed RMSE ({winner.RMSE:.3f} kW). The result should be interpreted alongside MAE and R², not as classification accuracy.")
        st.write("Limitations include a plant-specific historical dataset, data gaps that prevent some windows, the inclusion of nighttime zeros, and historical reanalysis/weather observations rather than future operational weather forecasts. Neither model is evidence of live forecasting capability or cross-plant generalisation.")

if __name__ == '__main__': main()

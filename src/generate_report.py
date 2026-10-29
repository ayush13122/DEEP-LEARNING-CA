"""Build the CA1 Word report after real training artifacts exist."""
from __future__ import annotations
import json
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'CA1_Deep_Learning_Solar_Forecasting_Report.docx'

def heading(doc, text, level=1):
    p=doc.add_heading(text, level=level); p.style.font.color.rgb=RGBColor(0,0,0); return p

def table(doc, headers, rows):
    t=doc.add_table(rows=1, cols=len(headers)); t.style='Table Grid'; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for cell, h in zip(t.rows[0].cells, headers):
        cell.text=str(h)
        for run in cell.paragraphs[0].runs: run.font.bold=True
    for row in rows:
        cells=t.add_row().cells
        for c,v in zip(cells,row): c.text=str(v)
    doc.add_paragraph()

def figure(doc, filename, caption):
    path=ROOT/'outputs'/'figures'/filename
    if path.exists():
        doc.add_picture(str(path), width=Inches(6.3)); p=doc.add_paragraph(caption); p.alignment=WD_ALIGN_PARAGRAPH.CENTER

def main():
    audit=json.loads((ROOT/'outputs'/'data_quality_audit.json').read_text())
    config=json.loads((ROOT/'artifacts'/'model_config.json').read_text())
    metrics=json.loads((ROOT/'outputs'/'model_metrics.json').read_text())
    doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.75); sec.bottom_margin=Inches(.75)
    styles=doc.styles; styles['Normal'].font.name='Aptos'; styles['Normal'].font.size=Pt(10.5)
    for name in ['Title','Heading 1','Heading 2']:
        styles[name].font.name='Aptos'; styles[name].font.color.rgb=RGBColor(0,0,0)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.space_after=Pt(20)
    r=p.add_run('Comparative Analysis of LSTM and 1D CNN for Model 2 Intraday Next Hour Solar Power Forecasting'); r.bold=True; r.font.size=Pt(20)
    for line in ['Deep Learning CA1 Project Report','Model 2 only  |  Historical retrospective evaluation','Prepared from the completed TensorFlow Keras experiment']:
        p=doc.add_paragraph(line); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()
    heading(doc,'Abstract')
    winner=min(['LSTM','1D-CNN'],key=lambda n:metrics[n]['RMSE'])
    doc.add_paragraph(f'This report evaluates an LSTM and a one-dimensional convolutional neural network for the Model 2 intraday next-hour solar-power task. Each forecast uses 24 consecutive hourly records ending at time t and predicts observed AC power at t+1. Dataset integrity checks confirmed {audit["rows_loaded"]:,} rows, no duplicates or missing cells, and exact one-hour target alignment. Windows crossing timestamp gaps were excluded. The final test period begins 1 January 2017 and ends with the available January 2018 observations. The lowest final-test RMSE among the two neural models was produced by {winner} ({metrics[winner]["RMSE"]:.3f} kW). Results are retrospective and do not claim live operational forecasting.')
    heading(doc,'Introduction and problem statement')
    doc.add_paragraph('Solar generation changes with solar geometry, clouds, temperature and the recent operating state of the plant. The practical question is whether information observed by the end of an hour can predict the next hour of AC power. This CA1 deliberately studies only the supplied Model 2 intraday problem and does not include Model 1 or LightGBM results.')
    heading(doc,'Objectives and scope')
    for s in ['Build a validated 24-hour multivariate sequence dataset without target leakage.','Compare compact TensorFlow Keras LSTM and 1D-CNN regressors on identical chronological test samples.','Report MAE, RMSE and R² in original AC-power units with a separate persistence sanity check.','Deliver a saved-artifact Streamlit dashboard and reproducible source code.']:
        doc.add_paragraph(s,style='List Bullet')
    heading(doc,'Dataset description and quality audit')
    doc.add_paragraph('The copied Model 2 dataset combines current observed AC power with plant telemetry, weather, cloud, solar-position and cyclical calendar variables. Raw timestamps are used only to validate continuity, construct sequences and assign chronological partitions; they are not model inputs.')
    table(doc,['Check','Verified result'],[['Rows and columns',f"{audit['rows_loaded']:,} × {audit['column_count']}"],['Model features',audit['feature_count']],['Feature time range',f"{audit['feature_timestamp_start']} to {audit['feature_timestamp_end']}"],['Duplicate timestamps',audit['duplicate_timestamps']],['Null cells',audit['null_cells']],['Irregular timestamp steps',audit['irregular_timestamp_steps']],['Invalid target alignments',audit['invalid_target_alignment_rows']],['Valid windows',audit['valid_sequences']],['Windows rejected for gaps',audit['rejected_for_timestamp_gaps']]])
    heading(doc,'Preprocessing and sequence construction')
    doc.add_paragraph('Rows are parsed and sorted chronologically. A candidate window is retained only when all 24 feature timestamps differ by exactly one hour and the final row has a verified target timestamp exactly one hour later. The window label is the final row’s target_ac_power. No target power, target timestamp or raw timestamp enters either neural model. Legitimate zero-power night observations are retained; gaps are not interpolated.')
    heading(doc,'Forecast definition')
    doc.add_paragraph('For a final input time t, X = [features(t−23), …, features(t)] and y = target_ac_power(t+1). Therefore the tensor shape is (samples, 24, 27), and the project is next-hour rather than 24-hour-ahead forecasting.')
    heading(doc,'Training validation and test methodology')
    split=config['split']; table(doc,['Partition','Target timestamps','Sequence samples'],[['Training',f"through {split['training_end']}",split['sample_counts']['train']],['Validation',f"{split['validation_start']} through {split['validation_end']}",split['sample_counts']['validation']],['Final test',f"from {split['test_start']} through {split['actual_ranges']['test']['end']}",split['sample_counts']['test']]])
    doc.add_paragraph('The split follows the existing Model 2 convention. Partitioning is based on target timestamps; a validation or test window may use its preceding historical context but never future labels. StandardScaler objects are fit only on training tensor rows and training labels, then predictions are inverse-transformed before metrics are calculated.')
    heading(doc,'Neural architectures and hyperparameters')
    table(doc,['Model','Architecture'],[['LSTM','Input 24×27 → LSTM 64 with dropout 0.15 → Dense 32 ReLU → Dropout 0.10 → linear output'],['1D CNN','Input 24×27 → Conv1D 32 causal → Conv1D 64 causal → global average pooling → Dense 32 ReLU → Dropout 0.15 → linear output']])
    doc.add_paragraph('Both models use Adam (learning rate 0.001), MSE loss, MAE monitoring, batch size 128, a maximum of 40 epochs, EarlyStopping with best-weight restoration, and ReduceLROnPlateau. The random seed is fixed at 42 where TensorFlow supports deterministic execution.')
    # Keep the complete metrics table together rather than splitting its final row across pages.
    doc.add_page_break()
    heading(doc,'Results and interpretation')
    table(doc,['Model','MAE kW','RMSE kW','R²'],[[name,f"{metrics[name]['MAE']:.3f}",f"{metrics[name]['RMSE']:.3f}",f"{metrics[name]['R2']:.4f}"] for name in ['LSTM','1D-CNN','Persistence']])
    doc.add_paragraph(f'All metrics use the same {metrics["test_sample_count"]:,} final-test timestamps and original AC-power units. {winner} has the lower neural-model RMSE. Persistence is reported separately; any performance claim must be read from this table rather than inferred from R² alone. R² is a variance-explained statistic, not per-record accuracy.')
    figure(doc,'actual_vs_predicted.png','Figure 1. Actual and neural-model predictions for an interpretable final-test slice.')
    figure(doc,'metric_comparison.png','Figure 2. Final-test RMSE comparison.')
    figure(doc,'residual_distribution.png','Figure 3. Final-test residual distributions.')
    figure(doc,'training_validation_loss.png','Figure 4. Training and validation MSE histories.')
    heading(doc,'Limitations and future scope')
    doc.add_paragraph('This result is plant-specific and historical. Its data end in January 2018; the environmental inputs are historical observations or reanalysis rather than future operational forecasts. Missing hours reduce valid sequences, nighttime zeros influence aggregate metrics, and the experiment does not establish cross-plant or real-time generalisation. Future work could add operational weather forecasts, live SCADA quality checks, uncertainty intervals, multi-horizon training and cross-plant validation.')
    heading(doc,'Conclusion')
    doc.add_paragraph(f'The project delivers a reproducible, leakage-aware comparison of LSTM and 1D-CNN next-hour solar forecasts using Model 2 alone. The reported final-test evidence identifies {winner} as the lower-RMSE neural model for this experiment, subject to the stated historical and plant-specific limitations.')
    heading(doc,'References')
    for ref in ['TensorFlow. Keras API documentation. https://www.tensorflow.org/api_docs/python/tf/keras','Keras. LSTM layer documentation. https://keras.io/api/layers/recurrent_layers/lstm/','Keras. Conv1D layer documentation. https://keras.io/api/layers/convolution_layers/convolution1d/','NREL. Photovoltaic Data Acquisition program. https://www.nrel.gov/pv/pvdaq.html','NASA POWER Project. https://power.larc.nasa.gov/']:
        doc.add_paragraph(ref,style='List Number')
    doc.save(OUT); print(OUT)

if __name__=='__main__': main()

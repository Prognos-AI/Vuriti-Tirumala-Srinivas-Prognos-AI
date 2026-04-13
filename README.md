# Vuriti-Tirumala-Srinivas-Prognos-AI

AI-driven predictive maintenance project for Remaining Useful Life (RUL) estimation using the NASA CMAPSS dataset.

## Project Structure

- `main.ipynb`: End-to-end notebook pipeline for data preparation, training, evaluation, and analysis.
- `dashboard.py`: Streamlit dashboard for model insights, metrics, and visual outputs.
- `requirements.txt`: Python dependency list.

### Data

- `data/train_FD001.txt`, `data/train_FD002.txt`, `data/train_FD003.txt`, `data/train_FD004.txt`
- `data/test_FD001.txt`, `data/test_FD002.txt`, `data/test_FD003.txt`, `data/test_FD004.txt`
- `data/RUL_FD001.txt`, `data/RUL_FD002.txt`, `data/RUL_FD003.txt`, `data/RUL_FD004.txt`

### Trained Models

- `models/gru_rul_model.keras`
- `models/lstm_rul_model.keras`
- `models/feature_scaler.pkl`
- `models/model_config.json`

### Generated Artifacts

- `artifacts/model_performance_report.json`
- `artifacts/model_summary_stats.json`
- `artifacts/per_dataset_metrics.csv`
- `artifacts/engine_risk.csv`
- `artifacts/test_predictions.csv`
- `artifacts/test_predictions_detailed.csv`

### Dashboard/Result Images

- `Output/Dashboard-GRU.png`
- `Output/Dashboard-LSTM.png`
- `Output/GRU-Data.png`
- `Output/LSTM-Data.png`
- `Output/GRU-Engine.png`
- `Output/LSTM-Engine.png`
- `Output/Metrics.png`
- `Output/Alert.png`

## Setup and Run

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook main.ipynb
```

4. Run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

## Notes

- The notebook produces model files and evaluation artifacts.
- The dashboard reads generated artifacts and model outputs to visualize performance and risk insights.

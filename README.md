# GSR Forecasting Modular File-by-File Colab Code

This package splits the full CNN-BiLSTM-STAM workflow into separate files for Google Colab and GitHub.

## Files

- `preprocessing.py` — data loading, cleaning, feature engineering, scaling, and 48-hour sequence construction
- `model.py` — STAM attention layer and CNN-BiLSTM-STAM model definition
- `train.py` — training and inverse prediction utilities
- `evaluate.py` — regression metrics and result saving
- `shap_analysis.py` — SHAP feature importance and temporal SHAP utilities
- `statistical_tests.py` — paired t-test, Wilcoxon, DM-HAC, bootstrap confidence interval
- `model_profile.py` — parameter count, model-size, and inference-time profiling
- `main_colab_runner.ipynb` — Colab notebook importing the files above

## Main dataset expected file

Upload your main dataset to Colab as:

```text
/content/complete_dataset.xlsx
```

Required columns:

```text
Timestamp
Air Temperature
Dew Point
Saturation Vapor Pressure
Relative Humidity
Wind Speed Avg
Wind Speed Instant
Wind Direction
GSR
```

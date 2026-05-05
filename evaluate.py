import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
    median_absolute_error,
    mean_absolute_percentage_error,
    max_error,
)
from scipy.stats import pearsonr, kendalltau, spearmanr


def smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-6, denom)
    return np.mean(2 * np.abs(y_pred - y_true) / denom) * 100


def mean_bias_error(y_true, y_pred):
    return np.mean(y_pred - y_true)


def calculate_metrics(y_true, y_pred, label="All test samples"):
    """Calculate regression metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask_nonzero = np.abs(y_true) > 1e-2
    y_true_masked = y_true[mask_nonzero]
    y_pred_masked = y_pred[mask_nonzero]

    return {
        "Evaluation": label,
        "N_test": len(y_true),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "EVS": explained_variance_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true_masked, y_pred_masked) * 100 if len(y_true_masked) else np.nan,
        "SMAPE": smape(y_true_masked, y_pred_masked) if len(y_true_masked) else np.nan,
        "Max_Error": max_error(y_true, y_pred),
        "Pearson_r": pearsonr(y_true, y_pred)[0],
        "Kendall_Tau": kendalltau(y_true, y_pred).correlation,
        "Spearman_Rho": spearmanr(y_true, y_pred).correlation,
        "MBE": mean_bias_error(y_true, y_pred),
        "RelRMSE_percent": 100 * np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(np.abs(y_true)),
    }


def print_metrics(metrics):
    print("\n--- REGRESSION METRICS ---")
    for k, v in metrics.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")


def save_metrics(metrics, output_csv="GSR_metrics.csv"):
    df = pd.DataFrame([metrics])
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to: {output_csv}")
    return df


def save_predictions(y_true, y_pred, output_csv="GSR_actual_vs_predicted.csv"):
    df_results = pd.DataFrame({
        "Actual_GSR": y_true.flatten(),
        "Predicted_GSR": y_pred.flatten(),
    })
    df_results.to_csv(output_csv, index=False)
    print(f"Saved actual and predicted GSR values to: {output_csv}")
    return df_results

# ==============================================================================
# metrics.py
# ==============================================================================
# Functions for computing and handling performance metrics.

from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from pathlib import Path
import unicodedata
import re

# COMPUTE ALL METRICS
# -----------------------------------------------------------------------------
# What this does:
# • Groups the long results table `df` by (model, horizon, output).
# • For each group, computes:
#     - RMSE, MAE, MBE (error stats) and R² (goodness of fit).
#     - Normalized versions nRMSE, nMAE, nMBE using mean(y_true) as the scale.
# • Returns a tidy `metrics_df` with one row per (model, horizon, output).
#
# Notes:
# • If mean(y_true) == 0, normalized metrics are set to NaN to avoid division by zero.
# • R² requires at least 2 points; otherwise it’s NaN.
# • Expects columns: ['model','horizon','output','y_true','y_pred'] in `df`.
def compute_all_metrics(df):
    """Computes a standard set of metrics (nRMSE, nMAE, nMBE, R2, etc.) from a results DataFrame."""
    print(">>> [Step 3/5] Computing performance metrics...")
    if df.empty:
        return pd.DataFrame(columns=["model", "horizon", "output", "nRMSE", "nMAE", "nMBE", "R2", "RMSE", "MAE", "MBE"])
    metrics_list = []
    # Group by model, horizon, and output variable to calculate metrics for each segment
    grouped = df.groupby(['model', 'horizon', 'output'])
    for (model, horizon, output), group in grouped:
        y_true = group['y_true'].values
        y_pred = group['y_pred'].values
        
        if len(y_true) == 0: continue

        error = y_pred - y_true
        mae = np.mean(np.abs(error))
        mbe = np.mean(error)
        rmse = np.sqrt(np.mean(error ** 2))
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
        
        mean_true = np.mean(y_true)
        # Calculate normalized metrics, handling division by zero
        nmae = mae / mean_true if mean_true != 0 else np.nan
        nmbe = mbe / mean_true if mean_true != 0 else np.nan
        nrmse = rmse / mean_true if mean_true != 0 else np.nan
        
        metrics_list.append([
            model, horizon, output,
            nrmse, nmae, nmbe, r2, rmse, mae, mbe
        ])
        
    metrics_df = pd.DataFrame(
        metrics_list,
        columns=["model", "horizon", "output", "nRMSE", "nMAE", "nMBE", "R2", "RMSE", "MAE", "MBE"]
    )
    print(f"    - Metrics computed for {len(metrics_df)} combinations.")
    print(">>> [Step 3/5] Metrics computation complete.\n")
    return metrics_df

def stack_external_metrics(forecast_dfs: dict[str, pd.DataFrame], model_name: str) -> pd.DataFrame:
    """Converts forecast DataFrames (like from Prophet) into a standardized metrics DataFrame."""
    print(f"    - Computing metrics for {model_name}...")
    frames = []
    _H_RE = re.compile(r"\d+") # Regex to find horizon number in column names like 'yhat24'
    
    for out_name, df in forecast_dfs.items():
        yhat_cols = [c for c in df.columns if str(c).lower().startswith("yhat")]
        
        for col in yhat_cols:
            m = _H_RE.findall(str(col))
            if not m: continue
            h = int(m[-1])

            sub = df[["y", col]].dropna()
            if sub.empty: continue
            
            y_true = sub["y"].to_numpy(float)
            y_pred = sub[col].to_numpy(float)
            
            # Re-use the metric calculation logic
            error = y_pred - y_true
            mae, mbe, rmse = np.mean(np.abs(error)), np.mean(error), np.sqrt(np.mean(error**2))
            r2 = r2_score(y_true, y_pred)
            mean_true = np.mean(y_true)
            
            nrmse = rmse / mean_true if mean_true != 0 else np.nan
            nmae = mae / mean_true if mean_true != 0 else np.nan
            nmbe = mbe / mean_true if mean_true != 0 else np.nan
            
            row = {
                "model": model_name, "horizon": h, "output": out_name,
                "nRMSE": nrmse, "nMAE": nmae, "nMBE": nmbe, "R2": r2, 
                "RMSE": rmse, "MAE": mae, "MBE": mbe
            }
            frames.append(row)
            
    print(f"    - {model_name} metrics finished.")
    return pd.DataFrame(frames)
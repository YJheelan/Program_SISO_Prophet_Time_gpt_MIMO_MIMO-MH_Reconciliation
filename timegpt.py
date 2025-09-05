# ==============================================================================
# timegpt.py
# ==============================================================================
# Placeholder for loading TimeGPT results.
import numpy as np
import pandas as pd
import os

def load_timegpt_metrics():
    """
    Load pre-computed TimeGPT metrics from a CSV file.

    Utility: This function loads a single metrics file for TimeGPT results. If the file is missing, it warns and returns None.

    Returns:
    - metrics_df: pandas DataFrame containing the metrics, or None if file not found.
    """
    file_path = "metrics_with_tgpt_8000S.csv"
    if os.path.exists(file_path):
        metrics_df = pd.read_csv(file_path)
    else:
        print(f"      - Warning: Metrics file not found at {file_path}")
        metrics_df = None
    return metrics_df


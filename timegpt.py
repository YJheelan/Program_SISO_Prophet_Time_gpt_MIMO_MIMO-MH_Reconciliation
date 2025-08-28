# ==============================================================================
# timegpt.py
# ==============================================================================
# Placeholder for loading TimeGPT results.
import numpy as np
import pandas as pd
import os


def load_timegpt_metrics():
    # Modified to load a single metrics CSV file instead of multiple forecast files
    file_path = "metrics_with_tgpt_8000S.csv"
    if os.path.exists(file_path):
        metrics_df = pd.read_csv(file_path)
    else:
        print(f"      - Warning: Metrics file not found at {file_path}")
        metrics_df = None
    return metrics_df


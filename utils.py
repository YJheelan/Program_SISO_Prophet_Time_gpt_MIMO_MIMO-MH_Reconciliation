# ==============================================================================
# utils.py
# ==============================================================================
# Utility functions for data loading and preprocessing.
import unicodedata
import re
import pandas as pd
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

def slugify(col: str) -> str:
    """Converts a string to a slug-like format (lowercase, underscores)."""
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode()
    col = re.sub(r"[^\w\s-]", "", col).strip().lower().replace(" ", "_")
    return col

def load_and_preprocess_data(
    file_path: Path,
    date_col: str = "Date",
    drop_policy: str = "mean", # "first", "last" or "mean"
) -> pd.DataFrame:
    """
    Load a CSV, normalize column names, create 'ds' from `date_col`,
    and deduplicate on 'ds' according to `drop_policy`.
    Returns a DataFrame with a unique, sorted 'ds' column.
    """
    print(">>> [Step 1/5] Loading and pre-processing data...")
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    # 1) normalize column names
    df.columns = [slugify(c) for c in df.columns]
    date_col = slugify(date_col)
    
    # 2) convert date → UTC
    df["ds"] = pd.to_datetime(df[date_col], utc=True)
    df = df.drop(columns=[date_col])
    
    # 3) handle duplicates on ds
    if drop_policy in ("first", "last"):
        before = len(df)
        df = df.drop_duplicates(subset="ds", keep=drop_policy)
        removed = before - len(df)
        if removed:
            print(f"    - {removed} duplicate(s) removed (keep='{drop_policy}').")
    elif drop_policy == "mean":
        df = df.groupby("ds", as_index=False).mean(numeric_only=True)
        print("    - Duplicates aggregated by taking the mean.")
    else:
        raise ValueError("drop_policy must be 'first', 'last' or 'mean'.")
    # 4) sort and reset index
    df = df.sort_values("ds").reset_index(drop=True)
    # 5) drop timezone if present
    if df["ds"].dt.tz is not None:
        df["ds"] = df["ds"].dt.tz_localize(None) # Make timezone-naive for compatibility
        
    print(f"    - Data loaded successfully: {len(df)} rows, {df.shape[1]} columns.")
    print(">>> [Step 1/5] Data loading complete.\n")
    return df

def prepare_data_mimo_mh(input_matrix, num_rows, max_horizon, window_size, num_outputs):
    """
    Build supervised learning matrices (X, Y) for MIMO multi-horizon training.

    SCENARIO
    --------
    • You have a multivariate time series in `input_matrix` with `num_rows` rows
      (time) and multiple columns (features/targets).
    • The first `num_outputs` columns correspond to the target variables
      you want to forecast (multi-output).
    • For each timestamp t, you want to predict the next `max_horizon` steps
      for those `num_outputs` targets.

    WINDOWING
    ---------
    • For each sample start t in [0, num_rows - window_size - max_horizon]:
        X_window = input_matrix[t : t + window_size, :].flatten()
        Y_window = input_matrix[t + window_size : t + window_size + max_horizon,
                                :num_outputs].flatten()
      X_window collects the *past* `window_size` frames (all features).
      Y_window concatenates the *future* `max_horizon` × `num_outputs`
      target block in time-major order.

    SHAPES
    ------
    • samples = num_rows - window_size - max_horizon + 1
    • X_windows.shape = (samples, window_size * n_features)
    • Y_windows.shape = (samples, max_horizon * num_outputs)

    Parameters
    ----------
    input_matrix : np.ndarray
        Full design matrix with time along axis 0. Must have at least `num_outputs`
        columns to slice target variables.
    num_rows : int
        Number of available time rows in `input_matrix`.
    max_horizon : int
        Maximum forecast horizon H (we will build targets for h = 1..H).
    window_size : int
        Number of lagged timesteps used as input context (sliding window length).
    num_outputs : int
        Number of target variables to forecast jointly.

    Returns
    -------
    X_windows, Y_windows : np.ndarray
        Flattened input windows and concatenated multi-horizon targets, ready for ELM.
    """
    samples = num_rows - window_size - max_horizon + 1
    # Input windows: [sample, flattened_window_features]
    X_windows = np.array([
        input_matrix[t : t + window_size].flatten()
        for t in range(samples)
    ])
    # Build Y as a concatenation of the future target block:
    # [ (t+1..t+H) × num_outputs ] flattened for each sample.
    Y_windows = np.array([
        input_matrix[t + window_size : t + window_size + max_horizon, :num_outputs].flatten()
        for t in range(samples)
    ])
    return X_windows, Y_windows
# ==============================================================================
# prophet.py
# ==============================================================================
# Functions to load and process Prophet model results.
import pandas as pd
from pathlib import Path

def load_prophet_forecasts(targets):
    """
    Loads pre-computed Prophet forecast files.

    Utility: Retrieves forecast CSVs for each target from a directory. Warns if files are missing.

    Arguments:
    - targets: list of str, target variable names.

    Returns:
    - forecast_dfs: dict of {target: DataFrame}, forecasts for each target.
    """
    print("    - Loading pre-computed Prophet forecasts...")
    # This assumes forecast files are stored in a specific directory
    base_path = Path("neuralprophet_results/forecasts/")
    forecast_dfs = {}
    
    for target in targets:
        file_path = base_path / f"{target}_forecast_last2y.csv"
        if file_path.exists():
            forecast_dfs[target] = pd.read_csv(file_path)
        else:
            print(f"      - Warning: Prophet forecast file not found for '{target}' at {file_path}")
            
    return forecast_dfs
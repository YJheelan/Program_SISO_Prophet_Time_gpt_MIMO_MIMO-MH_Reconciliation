# ==============================================================================
# main.py
# ==============================================================================

import pandas as pd
from pathlib import Path
import time
import datetime

# --- Import local modules ---
from utils import load_and_preprocess_data
from models import run_siso_all_horizons, run_mimo_all_horizons, run_mimo_mh_all_horizons
from metrics import compute_all_metrics, stack_external_metrics
from prophet import load_prophet_forecasts
from timegpt import load_timegpt_metrics
from plotting import plot_metrics_subplots, plot_normalized_metrics_subplots, plot_all_metrics_by_energy_variable, plot_metrics_by_variable_subplots
from menu import select_models

CONFIG = {
    "data_path": "Data.csv",
    "date_col": "Date",
    "split_date": '2021-01-01',
    "targets": [
        'production_totale_mw', 'thermique_mw', 'hydraulique_mw', 
        'micro-hydraulique_mw', 'solaire_photovoltaique_mw', 'eolien_mw', 
        'bioenergies_mw', 'importations_mw'
    ],
    "elm_params": {
        "n_hidden": 1000,
        "lambda_reg": 1e-6,
        "n_init": 1,
    },
    "window_params": {
        "n_lags": 48,       # Input window size
        "n_forecasts": 24,  # Max horizon
    },
}
# ==============================================================================
# Helper functions for main execution
# ==============================================================================
def get_user_input(prompt, default_value, target_type):
    """Generic function to get user input with a default value and type casting."""
    while True:
        user_str = input(f"{prompt} [default: {default_value}] (press Enter to use default): ")
        if not user_str:
            return default_value
        try:
            return target_type(user_str)
        except (ValueError, TypeError):
            print(f"Invalid input. Please enter a value of type '{target_type.__name__}'.")

def get_hyperparameters(config):
    """Prompts the user to set hyperparameters, with defaults."""
    print("\n" + "="*60)
    print("HYPERPARAMETER CONFIGURATION")
    print("="*60)
    
    # Get Split Date
    while True:
        date_str = get_user_input(
            "Enter the split date (YYYY-MM-DD)",
            config["split_date"],
            str
        )
        try:
            # Validate date format
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
            config["split_date"] = date_str
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    # Get ELM parameters
    config["elm_params"]["lambda_reg"] = get_user_input(
        "Enter the regularization lambda for ELM (e.g., 1e-6)",
        config["elm_params"]["lambda_reg"],
        float
    )
    config["elm_params"]["n_hidden"] = get_user_input(
        "Enter the number of hidden neurons for ELM",
        config["elm_params"]["n_hidden"],
        int
    )
    config["elm_params"]["n_init"] = get_user_input(
        "Enter the number of initializations for ELM",
        config["elm_params"]["n_init"],
        int
    )
    
    print("="*60)
    print("Final configuration used:")
    print(f"  - Split date: {config['split_date']}")
    print(f"  - Lambda: {config['elm_params']['lambda_reg']}")
    print(f"  - Hidden neurons: {config['elm_params']['n_hidden']}")
    print(f"  - Initializations: {config['elm_params']['n_init']}")
    print("="*60 + "\n")
    
    return config
        
def process_results(df_results, base_model_name, all_results_list, reconcile):
    """Processes the results dataframe, adding either base or reconciled based on reconcile flag."""
    if reconcile:
        # Only add reconciled if reconciliation is applied
        if 'y_pred_rec' in df_results.columns:
            df_rec = df_results[['datetime', 'horizon', 'output', 'y_true', 'y_pred_rec']].copy()
            df_rec.rename(columns={'y_pred_rec': 'y_pred'}, inplace=True)
            df_rec['model'] = f"{base_model_name}-REC"
            all_results_list.append(df_rec)
    else:
        # Add base model
        df_base = df_results[['datetime', 'horizon', 'output', 'y_true', 'y_pred']].copy()
        df_base['model'] = base_model_name
        all_results_list.append(df_base)

# ==============================================================================
# Main execution block
# ==============================================================================
def main():
    import os
    # folder of figures
    os.makedirs("figures", exist_ok=True)
    """Main function to run the forecasting pipeline."""
    
    # Step 0: Get user choices for models and hyperparameters
    models_to_run = select_models()
    if not models_to_run:
        print("No model selected. Exiting program.")
        return
        
    # Ask for hyperparameters only if an ELM model is selected
    if any(m[0] in {'siso', 'mimo', 'mimo-mh'} for m in models_to_run):
        final_config = get_hyperparameters(CONFIG)
    else:
        final_config = CONFIG # Use default config if no ELM models are run
        
    models_to_plot = []
    for model_name, reconcile_flag in models_to_run:
        if model_name == 'prophet':
            models_to_plot.append('Prophet') # On force la casse exacte
        elif model_name == 'timegpt':
            models_to_plot.append('TimeGPT') # On force la casse exacte
        else:
            suffix = '-REC' if reconcile_flag else ''
            models_to_plot.append(f"{model_name.upper()}{suffix}")
    
    # Step 1: Load and preprocess data
    df_all = load_and_preprocess_data(Path(final_config["data_path"]), final_config["date_col"])
    df_subset = df_all[["ds"] + final_config["targets"]]

    print(">>> [Step 2/5] Running forecasting models...")
    all_results = []

    # Run selected models
    for model_name, reconcile_flag in models_to_run:
        start_time = time.time()
        
        if model_name == 'siso':
            df_res = run_siso_all_horizons(
                df_subset, final_config["split_date"], final_config["targets"], 
                final_config["window_params"]["n_forecasts"], final_config["window_params"]["n_lags"], 
                final_config["elm_params"], reconcile=reconcile_flag
            )
            process_results(df_res, 'SISO', all_results, reconcile_flag)
            
        elif model_name == 'mimo':
            df_res = run_mimo_all_horizons(
                df_subset, final_config["split_date"], final_config["targets"], 
                final_config["window_params"]["n_forecasts"], final_config["window_params"]["n_lags"], 
                final_config["elm_params"], reconcile=reconcile_flag
            )
            process_results(df_res, 'MIMO', all_results, reconcile_flag)
            
        elif model_name == 'mimo-mh':
            df_res = run_mimo_mh_all_horizons(
                df_subset, final_config["split_date"], final_config["targets"], 
                final_config["window_params"]["n_forecasts"], final_config["window_params"]["n_lags"], 
                final_config["elm_params"], reconcile=reconcile_flag
            )
            process_results(df_res, 'MIMO-MH', all_results, reconcile_flag)

        # External models are handled in Step 4, so no action here
        elif model_name in ['prophet', 'timegpt']:
            continue
            
        end_time = time.time()
        print(f"    -> Execution time: {end_time - start_time:.2f} seconds.")

    df_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    print(">>> [Step 2/5] Forecasting models finished.\n")

    # Step 3: Compute metrics for our models
    metrics_df = compute_all_metrics(df_results)
    
    # Step 4: Load and compute metrics for external models
    print(">>> [Step 4/5] Loading and processing external model results...")
    all_metrics = [metrics_df]
    
    selected_model_names = [m[0] for m in models_to_run]
    
    if 'prophet' in selected_model_names:
        start_time = time.time()
        prophet_forecasts = load_prophet_forecasts(final_config["targets"])
        prophet_metrics = stack_external_metrics(prophet_forecasts, model_name="Prophet")
        if not prophet_metrics.empty:
            all_metrics.append(prophet_metrics)
        end_time = time.time()
        print(f"Prophet results processed in {end_time - start_time:.2f} seconds.")

    if 'timegpt' in selected_model_names:
        start_time = time.time()
        timegpt_metrics = load_timegpt_metrics()
        if timegpt_metrics is not None:
            all_metrics.append(timegpt_metrics)
        end_time = time.time()
        print(f"TimeGPT results processed in {end_time - start_time:.2f} seconds.")
    
    # Combine all metrics into a single DataFrame
    final_metrics = pd.concat(all_metrics, ignore_index=True).sort_values(
        ["output", "horizon", "model"]
    ).reset_index(drop=True)

    print("    - All model metrics have been combined.")
    print(">>> [Step 4/5] External models processed.\n")

    # Save metrics to CSV
    output_filename = "final_metrics_comparison.csv"
    final_metrics.to_csv(output_filename, index=False)
    print(f"    - Final metrics saved to '{output_filename}'")
    
    # Print the final metrics in a clear way
    print("\n>>> Final Metrics Comparison:")
    metrics_to_print = final_metrics[final_metrics['model'].isin(models_to_plot)]
    if not metrics_to_print.empty:
        print(metrics_to_print.to_string(index=False))
    else:
        print("No metrics to display for the selected models.")
    print("\n")
    
    # Step 5: Plot results if there are any
    if not metrics_to_print.empty:
        plot_metrics_subplots(final_metrics, models_to_plot=models_to_plot)
        plot_normalized_metrics_subplots(final_metrics, models_to_plot=models_to_plot)
        plot_all_metrics_by_energy_variable(final_metrics, models_to_plot=models_to_plot)
        plot_metrics_by_variable_subplots(final_metrics, models_to_plot=models_to_plot) # Call to the new function
    else:
        print(">>> [Step 5/5] Skipping plot generation as no results were produced.")

if __name__ == '__main__':
    main()
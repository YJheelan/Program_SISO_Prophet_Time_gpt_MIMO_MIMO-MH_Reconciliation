# ==============================================================================
# plotting.py
# ==============================================================================
# Functions for visualizing model performance.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams
import numpy as np


def plot_metrics_subplots(metrics_df, models_to_plot=None):
    """
    Plots nRMSE, R2, MAE, and MBE in a 2x2 subplot figure.
    Each subplot compares the performance of different models across the forecast horizon.
    """
    print(">>> [Step 5/5] Generating plots...")
    print("    - Generating Plot 1: Key Metrics Subplots (RMSE, R2, MAE, MBE)...")
    
    if models_to_plot:
        metrics_df = metrics_df[metrics_df["model"].isin(models_to_plot)].copy()

    models = sorted(metrics_df["model"].unique())
    markers = ['o', 'x', 's', '^', 'd', 'v']
    marker_map = {model: markers[i % len(markers)] for i, model in enumerate(models)}

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Model Performance Comparison Across Horizons", fontsize=16)
    
    metric_names = ["RMSE", "R2", "MAE", "MBE"]
    
    for ax, metric in zip(axs.flat, metric_names):
        for model in models:
            # Calculate the mean metric value across all outputs for each horizon
            data = (metrics_df[metrics_df["model"] == model]
                    .groupby("horizon", observed=True)[metric]
                    .mean())
            ax.plot(data.index, data.values, marker=marker_map[model], linestyle='-', label=model)
        
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
    
    # Common X-axis label
    axs[1, 0].set_xlabel("Forecast Horizon (hours)")
    axs[1, 1].set_xlabel("Forecast Horizon (hours)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_normalized_metrics_subplots(metrics_df, models_to_plot=None):
    """
    Plots nRMSE, R2, nMAE, and nMBE in a 2x2 subplot figure.
    Each subplot compares the performance of different models across the forecast horizon.
    """
    print("    - Generating Plot 2: Normalized Key Metrics Subplots (nRMSE, R2, nMAE, nMBE)...")
    
    if models_to_plot:
        metrics_df = metrics_df[metrics_df["model"].isin(models_to_plot)].copy()

    models = sorted(metrics_df["model"].unique())
    markers = ['o', 'x', 's', '^', 'd', 'v']
    marker_map = {model: markers[i % len(markers)] for i, model in enumerate(models)}

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Model Performance Comparison Across Horizons (Normalized)", fontsize=16)
    
    metric_names = ["nRMSE", "R2", "nMAE", "nMBE"]
    
    for ax, metric in zip(axs.flat, metric_names):
        for model in models:
            # Calculate the mean metric value across all outputs for each horizon
            data = (metrics_df[metrics_df["model"] == model]
                    .groupby("horizon", observed=True)[metric]
                    .mean())
            ax.plot(data.index, data.values, marker=marker_map[model], linestyle='-', label=model)
        
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
    
    # Common X-axis label
    axs[1, 0].set_xlabel("Forecast Horizon (hours)")
    axs[1, 1].set_xlabel("Forecast Horizon (hours)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Savefig
    filename = f"figures/model_comparison_normalized_metrics.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {filename}")
    
    plt.show()

def plot_all_metrics_by_energy_variable(metrics_df, models_to_plot=None):
    """
    Plots nRMSE, nMAE, nMBE, and R2 for each output variable on separate large plots.
    On each plot, colors represent the output, while line styles/markers represent the model.
    """
    print("    - Generating Plot 3: Metrics by Energy Variable (Combined View)...")

    if models_to_plot:
        metrics_df = metrics_df[metrics_df["model"].isin(models_to_plot)].copy()

    outputs = sorted(metrics_df["output"].unique())
    models = sorted(metrics_df["model"].unique())
    
    # Define consistent styling for models
    styles = ['-', '--', ':', '-.']
    markers = ['o', 'x', 's', '^', 'd', 'v', '<', '>']
    linestyle_by_model = {model: styles[i % len(styles)] for i, model in enumerate(models)}
    marker_by_model = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    
    # Use matplotlib's color cycle for variables
    prop_cycle = rcParams.get("axes.prop_cycle", plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, len(outputs)))))
    color_iterator = iter(prop_cycle)
    color_map = {out: next(color_iterator)['color'] for out in outputs}

    # List of metrics to plot
    metrics_to_plot = ["nRMSE", "nMAE", "nMBE", "R2"]

    # Loop over each metric to create a separate plot
    for metric in metrics_to_plot:
        print(f"      - Plotting {metric}...")
        fig, ax = plt.subplots(figsize=(14, 9))
        
        for out in outputs:
            for model in models:
                data = (metrics_df[(metrics_df["output"] == out) & (metrics_df["model"] == model)]
                        .sort_values("horizon"))
                if data.empty:
                    continue
                
                ax.plot(
                    data["horizon"], data[metric],  # Use the current metric for y-data
                    label=f"{out} ({model})",
                    linestyle=linestyle_by_model[model],
                    marker=marker_by_model[model],
                    color=color_map[out],
                    alpha=0.8
                )

        ax.set_xlabel("Forecast Horizon (hours)")
        ax.set_ylabel(metric) # Update Y-axis label
        ax.set_title(f"{metric} vs. Horizon by Energy Variable and Model") # Update title
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Create custom legends for clarity
        output_handles = [Line2D([0], [0], color=color_map[o], lw=2, label=o) for o in outputs]
        model_handles = [Line2D([0], [0], color='gray', lw=2, linestyle=linestyle_by_model[m], marker=marker_by_model[m], label=m) for m in models]
        
        # Place legends outside the plot to minimize overlap
        legend1 = ax.legend(handles=output_handles, title="Energy Variable", loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.add_artist(legend1)
        ax.legend(handles=model_handles, title="Model", loc="lower left", bbox_to_anchor=(1.02, 0))
        
        # Adjust layout to make room for the external legends
        fig.tight_layout(rect=[0, 0, 0.85, 1]) 

        # Savefig
        filename = f"figures/model_comparison_{metric.lower()}_by_variable.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
        
        plt.show()

def plot_metrics_by_variable_subplots(metrics_df, models_to_plot=None):
    """
    Creates a grid of subplots for each metric. Each subplot shows the model comparison for a single energy variable.
    """
    print("    - Generating Plot 4: Metrics by Energy Variable (Subplot View)...")

    if models_to_plot:
        metrics_df = metrics_df[metrics_df["model"].isin(models_to_plot)].copy()

    outputs = sorted(metrics_df["output"].unique())
    models = sorted(metrics_df["model"].unique())
    metrics_to_plot = ["nRMSE", "R2", "nMAE", "nMBE"]
    
    markers = ['o', 'x', 's', '^', 'd', 'v']
    marker_map = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    
    # Determine grid size
    n_outputs = len(outputs)
    n_cols = 2
    n_rows = (n_outputs + n_cols - 1) // n_cols

    for metric in metrics_to_plot:
        print(f"      - Plotting {metric}...")
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
        axes = axes.flatten()
        
        fig.suptitle(f"{metric} Comparison Across Models by Energy Variable", fontsize=16, y=1.02)

        for i, var in enumerate(outputs):
            ax = axes[i]
            for model in models:
                subset = metrics_df[(metrics_df['output'] == var) & (metrics_df['model'] == model)]
                if not subset.empty:
                    ax.plot(subset['horizon'], subset[metric], marker=marker_map[model], linestyle='-', label=model)
            
            ax.set_title(var)
            ax.set_ylabel(metric)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()

        # Add common X-axis labels to the bottom plots
        for i in range(n_rows * n_cols - n_cols, n_rows * n_cols):
             if i < n_outputs:
                axes[i].set_xlabel("Forecast Horizon (hours)")

        # Hide any unused subplots
        for i in range(n_outputs, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Savefig
        filename = f"figures/{metric.lower()}_by_variable_subplots.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")

        plt.show()

    print(">>> [Step 5/5] Plot generation complete.\n")
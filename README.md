# Multi-Source Energy Forecasting with Optimal Reconciliation via Multi-Input Multi-Output Multi-Horizon (MIMO-MH) and Extreme Learning Machine (ELM)
## Overview
The intermittent nature of renewable energy sources poses several challenges, particularly regarding reliability, energy quality, and supply-demand balance [[1]](#ref1). In this context, forecasting electricity production from renewable sources such as wind and solar energy becomes essential for the efficient and continuous operation of the electrical grid [[2]](#ref2).

The Multi-Input Multi-Output Multi-Horizon (MIMO-MH) approach with reconciliation and Extreme Learning Machine (ELM) enables synchronization of forecasts from different sources (solar, thermal, hydraulic, imports, etc.) to ensure consistency with net consumption (equivalent to grid demand), while capturing global physical interactions and constraints (supply-demand balance, import/export, self-consumption) [[3]](#ref3).

Unlike Single-Input Single-Output (SISO) models, which process each source independently, MIMO leverages correlations between sources and shared variability, thereby improving aggregate accuracy and reducing total deviations (through the compensatory effect between errors). In parallel, ELM offers fast learning, an analytical closed-form solution, and low computational load, making it ideal for near real-time adaptation. This approach optimizes the forecasting of final demand (net consumption), which is essential for dispatching, import management, and grid stability, while providing a robust and cost-effective solution for highly variable, self-consumption-prone multi-energy systems.

## Flowchart

flowchart TD
    A["main.py: Entry Point - Load Config & Hyperparams"] --> B["menu.py: select_models() - User Interactive Menu (1-14 Options)"]
    B --> C["utils.py: load_and_preprocess_data() - Load CSV, Slugify, English Rename, Handle Duplicates, Create 'ds'"]
    C --> D["models.py: Run Selected Models Loop - Branch by Type"]
    D --> E{Model}
    E -->|"SISO/MIMO/MIMO-MH"| F["models.py: run_siso_all_horizons() / run_mimo_all_horizons() / run_mimo_mh_all_horizons() - Sliding Windows, Train per Var/Horizon or Joint"]
    E -->|"Persistence (Optional)"| G["models.py: run_persistence_models() - Horizon & 24h Baselines, Compute get_metrics() per Output"]
    E -->|"Prophet"| H["prophet.py: load_prophet_forecasts() - Load Pre-Computed CSVs by Target"]
    E -->|"TimeGPT"| I["timegpt.py: load_timegpt_metrics() - Load Metrics CSV or Warn None"]
    F --> J["elm.py: train_elm() & predict_elm() - Random ReLU Hidden, Ridge Readout, Non-Neg Predictions"]
    J --> K["reconciliation.py: Optional (if REC Flag) - create_S_from_outputs(), estimate_W_from_train_residuals(), reconcile_all_horizons() (WLS/MinT, Coherence Check)"]
    K --> L["main.py: process_results() - Extract y_true/y_pred (or REC), Add 'model' Column, Build df_results"]
    G --> M["metrics.py: get_metrics() - Direct Baselines to df_results"]
    H --> N["metrics.py: stack_external_metrics() - Regex yhat_h, Compute Metrics per Horizon/Output"]
    I --> N
    M --> O["metrics.py: compute_all_metrics(df_results) - Group & Calc RMSE/MAE/MBE/R2/nRMSE/etc. per Model/Horizon/Output"]
    N --> O
    L --> O
    O --> P["metrics.py: compute_correlations() (Output/Horizon Spearman) & compute_pacf_matrix() (Errors PACF)"]
    P --> Q["plotting.py: Auto Plots - plot_metrics_subplots(), plot_normalized_metrics_subplots(), plot_all_metrics_by_energy_variable(), plot_metrics_by_variable_subplots(), plot_reconciliation_improvement() (% nRMSE Gain), plot_model_times() (Bar), plot_correlations() (Heatmaps), plot_pacf_solar()"]


    style A fill:#e1f5fe
    style Q fill:#e8f5e8


## Dataset Description

The performance of energy forecasting models directly depends on the quality and representativeness of the data. To capture the dynamics of interactions between various energy sources, we use detailed hourly time series over a sufficiently long period to reflect actual variability. A time series is a sequence of observations indexed by time. In this report, the time series represent the hourly electricity production in MWh from different sources (Thermal, Hydropower, Micro-hydro, Solar PV, Wind, Bioenergy, Imports). They also include the average production cost in €/MWh and the total production in MWh. These series are managed by EDF for the Corsica region (https://opendata-corse.edf.fr/pages/home0/), covering the period from 2016 to 2022 with hourly resolution, ensuring both reliability and representativeness of the Corsican island energy context.

Column names in the processed data :
- `total_production_mw`: Total electricity production.
- `thermal_mw`: Thermal energy production.
- `hydraulic_mw`: Hydraulic energy production.
- `micro_hydraulic_mw`: Micro-hydraulic energy production.
- `solar_photovoltaic_mw`: Solar photovoltaic energy production.
- `wind_mw`: Wind energy production.
- `bioenergy_mw`: Bioenergy production.
- `imports_mw`: Imported energy.

## Project Structure

### Core Files and Modules

| File/Module | Description |
|-------------|-------------|
| `main.py` | Main entry point that loads data, runs selected models via an interactive menu, computes metrics, and generates plots. |
| `utils.py` | Utility functions for data loading, preprocessing (e.g., `load_and_preprocess_data` for loading CSV, normalizing columns, handling duplicates; `prepare_data_mimo_mh` for creating sliding windows for MIMO-MH). |
| `elm.py` | Extreme Learning Machine implementation (e.g., `train_elm` for training with random hidden layers and ridge regression; `predict_elm` for predictions). |
| `models.py` | Core forecasting models (e.g., `run_siso_all_horizons` for SISO; `run_mimo_all_horizons` for MIMO; `run_mimo_mh_all_horizons` for MIMO-MH; `run_persistence_models` for baseline persistence). Each function takes data, split date, output names, horizons, window size, ELM params, and optional reconciliation flag; returns a DataFrame with true and predicted values. |
| `reconciliation.py` | Hierarchical reconciliation functions (e.g., `create_S_from_outputs` builds aggregation matrix; `estimate_W_from_train_residuals` estimates error covariance; `reconcile_all_horizons` applies WLS reconciliation). |
| `metrics.py` | Metric computation (e.g., `compute_all_metrics` computes nRMSE, nMAE, etc., from results; `get_metrics` for individual calculations; `stack_external_metrics` for external models). Also includes correlation and PACF computations. |
| `plotting.py` | Visualization functions (e.g., `plot_normalized_metrics_subplots` for metric plots; `plot_correlations` for Spearman correlations; `plot_pacf_solar` for partial autocorrelation). |
| `menu.py` | Interactive menu for model selection (e.g., `select_models` returns list of models and reconciliation flags). |
| `prophet.py` | Loads pre-computed Prophet forecasts (e.g., `load_prophet_forecasts`). |
| `timegpt.py` | Loads pre-computed TimeGPT metrics (e.g., `load_timegpt_metrics`). |

### Forecasting Models

| Module | Model Type | Description |
|--------|------------|-------------|
| `models.py` | SISO | Independent forecasting per source and horizon |
| `models.py` | MIMO | Joint multi-source forecasting per horizon |
| `models.py` | MIMO-MH | Joint multi-source, multi-horizon forecasting |
| `reconciliation.py` | SISO/MIMO/MIMO-MH with REC | Post-processing reconciliation for coherence (MinT/WLS) |
| `prophet.py` | Prophet | External univariate time-series model benchmark |
| `timegpt.py` | TimeGPT | External foundation model benchmark for metrics |

## Methodology

### 1. Single Input Single Output (SISO)
Based on the Extreme Learning Machine (ELM): a single hidden layer neural network with very fast learning and low computational cost, suitable for near real-time forecasting.
Forecasts a single energy source for a given horizon independently.
Provides good accuracy but remains limited in scope and slower when scaling to multiple sources and horizons.

### 2. Multi Input Multi Output (MIMO)
Jointly forecasts all energy sources for a given horizon.
Exploits cross-source correlations and shared variability patterns to improve robustness and overall aggregate performance.
Extends beyond SISO by integrating interdependencies between sources

### 3. Multi-Horizon Extension (MIMO-MH)
Generalizes MIMO to predict all sources across multiple horizons simultaneously within a single model.
Increases efficiency and ensures temporal consistency across forecasts.
Followed by an optimal reconciliation step, ensuring that the sum of individual source forecasts always matches the expected system-level total at each horizon.

### 4. Hierarchical Forecast Reconciliation
Applies the Minimum Trace (MinT) reconciliation method with Weighted Least Squares (WLS).
Ensures that the sum of disaggregated forecasts (e.g., by source) always matches the corresponding aggregate totals at each horizon.
Minimizes the overall forecast error variance, improving coherence, accuracy, and consistency across all hierarchical levels.

### 5. Benchmarking Methods
- **ELM Variants**: Base and reconciled (REC) versions of SISO, MIMO, MIMO-MH.
- **External Models**: Prophet (probabilistic univariate) and TimeGPT (pre-trained foundation model).
- **Persistence Models**: Simple baselines using historical values (last value at horizon or 24h back). Implemented in `models.py` as `run_persistence_models`.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Complete Installation
For a single command installation of all required packages:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Usage

#### Quick Start
```bash
python main.py
```
or with
```bash
python Run_program.ipynb
```

#### Path Configuration
Ensure the script runs from the project root.

#### Interactive Execution Pipeline

The main script provides an interactive menu-driven interface for executing various model combinations. Upon launch, users can select from predefined execution scenarios or create custom model combinations. Hyperparameters (e.g., split date, ELM hidden neurons) are prompted if ELM models are selected.

#### Execution Workflow

**1. Interactive Menu System**
   - Displays comprehensive model execution options
   - Supports flexible model selection through numbered choices
   - Includes custom execution mode for advanced users

**2. Automated Data Pipeline**
   - Loads energy time-series data using load_and_preprocess_data() from utils.py (arguments: file_path, date_col, drop_policy; returns: preprocessed DataFrame with 'ds' timestamp and English-renamed columns).
   - Preprocesses data with feature engineering, normalization, and column renaming to English.
   - Creates sliding window matrices optimized for different model architectures using prepare_data_mimo_mh() (arguments: input_matrix, num_rows, max_horizon, window_size, num_outputs; returns: X_windows, Y_windows arrays).
   - Initializes result containers for performance tracking

**3. Execution Options**

| Option | Models Executed | Description |
|--------|-----------------|-------------|
| **1** | SISO | Run SISO (base) : Independent per-source forecasting |
| **2** | MIMO | Run MIMO (base) : Joint multi-source per horizon |
| **3** | MIMO-MH | Run MIMO-MH (base) : Joint multi-source multi-horizon |
| **4** | SISO-REC | Run SISO with reconciliation |
| **5** | MIMO-REC | Run MIMO with reconciliation |
| **6** | MIMO-MH-REC | Run MIMO-MH with reconciliation |
| **7** | SISO (Base + REC) | Run SISO base and reconciled |
| **8** | MIMO (Base + REC) | Run MIMO base and reconciled |
| **9** | MIMO-MH (Base + REC) | Run MIMO-MH base and reconciled |
| **10** | ALL models (Base + REC) | Run all SISO/MIMO/MIMO-MH base and reconciled |
| **11** | Prophet + TimeGPT | Run external models |
| **12** | ALL models (ELM Base + External) | Run all base model + Prophet + TimeGPT |
| **13** | ALL models (ELM REC + External) | Run all reconciled model + Prophet + TimeGPT | 
| **14** | Custom Selection | User-defined model combinations (with options for base/reconciled/both for ELM models) |

#### Output Generation


The pipeline automatically generates:
- Model performance metrics and comparison tables (saved to final_metrics_comparison.csv)
- Forecast visualizations (saved to figures/ directory, e.g., normalized metrics subplots, correlations)
- Execution logs and timing statistics
- Spearman correlation matrices (between outputs or horizons) and PACF plots for analysis
- Reconciliation improvement plots showing % reduction in nRMSE due to reconciliation.
- Model execution time bar charts.

## Key Features

- Computational Efficiency: ELM provides fast training with analytical solutions (see train_elm in elm.py: arguments include X_train, Y_train, num_hidden, etc.; returns best model tuple).
- Physical Constraints: Maintains energy balance and grid stability requirements via reconciliation.
- Scalability: Handles multiple energy sources simultaneously.
- Real-time Capability: Suitable for operational forecasting systems.
- Robustness: Handles high variability and intermittency of renewable sources.
- Metrics and Visualizations: Computes normalized errors (nRMSE, etc.) and generates comparative plots (see plotting.py functions).
- External Integration: Supports loading results from Prophet and TimeGPT for benchmarking.

## References
**<a id="ref1">[1]</a>** Sheraz Aslam, Herodotos Herodotou, Syed Muhammad Mohsin, Nadeem Javaid, Nouman Ashraf, and Shahzad Aslam. [A survey on deep learning methods for power load and renewable energy forecasting in smart microgrids](https://doi.org/10.1016/j.rser.2021.110992). Renewable and Sustainable Energy Reviews, 144:110992, 2021.

**<a id="ref2">[2]</a>** Gilles Notton, Marie-Laure Nivet, Cyril Voyant, Christophe Paoli, Christophe Darras, Fabrice Motte, and Alexis Fouilloy. [Intermittent and stochastic character of renewable energy sources: Consequences, cost of intermittence and benefit of forecasting](https://doi.org/10.1016/j.rser.2018.02.007). Renewable and Sustainable Energy Reviews, 87:96–105, 2018.

**<a id="ref3">[3]</a>** Cyril Voyant, Milan Despotovic, Gilles Notton, Yves-Marie Saint-Drenan, Mohammed Asloune, and Luis Garcia-Gutierrez. [On the importance of clear-sky model in short-term solar radiation forecasting](https://doi.org/10.1016/j.solener.2025.113490). Solar Energy, 294:113490, 2025.


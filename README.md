# Multi-Source Energy Forecasting with Optimal Reconciliation via Multi-Input Multi-Output Multi-Horizon (MIMO-MH) and Extreme Learning Machine (ELM)
## Overview
The intermittent nature of renewable energy sources poses several challenges, particularly regarding reliability, energy quality, and supply-demand balance [[1]](#ref1). In this context, forecasting electricity production from renewable sources such as wind and solar energy becomes essential for the efficient and continuous operation of the electrical grid [[2]](#ref2).

The Multi-Input Multi-Output Multi-Horizon (MIMO-MH) approach with reconciliation and Extreme Learning Machine (ELM) enables synchronization of forecasts from different sources (solar, thermal, hydraulic, imports, etc.) to ensure consistency with net consumption (equivalent to grid demand), while capturing global physical interactions and constraints (supply-demand balance, import/export, self-consumption) [[3]](#ref3).

Unlike Single-Input Single-Output (SISO) models, which process each source independently, MIMO leverages correlations between sources and shared variability, thereby improving aggregate accuracy and reducing total deviations (through the compensatory effect between errors). In parallel, ELM offers fast learning, an analytical closed-form solution, and low computational load, making it ideal for near real-time adaptation. This approach optimizes the forecasting of final demand (net consumption), which is essential for dispatching, import management, and grid stability, while providing a robust and cost-effective solution for highly variable, self-consumption-prone multi-energy systems.

## Dataset Description

The performance of energy forecasting models directly depends on the quality and representativeness of the data. To capture the dynamics of interactions between various energy sources, we use detailed hourly time series over a sufficiently long period to reflect actual variability. A time series is a sequence of observations indexed by time. In this report, the time series represent the hourly electricity production in MWh from different sources (Thermal, Hydropower, Micro-hydro, Solar PV, Wind, Bioenergy, Imports). They also include the average production cost in €/MWh and the total production in MWh. These series are managed by EDF for the Corsica region (https://opendata-corse.edf.fr/pages/home0/), covering the period from 2016 to 2022 with hourly resolution, ensuring both reliability and representativeness of the Corsican island energy context.

## Project Structure

### Core Modules

| Module | Description |
|--------|-------------|
| `config.py` | (Implicit in main.py) Global settings, hyperparameters, and paths |
| `utils.py` | Data loading, preprocessing, and MIMO-MH window preparation |
| `elm.py` | ELM training and prediction with ridge regularization |
| `models.py` | Core forecasting runners for SISO, MIMO, MIMO-MH with optional reconciliation |
| `reconciliation.py` | Hierarchical reconciliation using MinT/WLS for forecast coherence |
| `metrics.py` | Computation of performance metrics (nRMSE, R2, nMAE, nMBE, etc.) |
| `plotting.py` | Visualization functions for metrics comparison and subplots |
| `prophet.py` | Loading pre-computed Prophet forecasts |
| `timegpt.py` | Loading TimeGPT metrics |
| `menu.py` | Interactive menu for model selection |
| `main.py` | Orchestrates the pipeline: data loading, model running, metrics, plotting |

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
Forecasts each energy source and horizon independently using ELM, without leveraging cross-source correlations.

### 2. Multi Input Multi Output (MIMO)
Jointly forecasts multiple sources per horizon, capturing interdependencies and shared patterns for better aggregate performance.

### 3. Multi-Horizon Extension (MIMO-MH)
Extends MIMO to predict multiple horizons simultaneously in a single model, improving efficiency and temporal consistency.

### 4. Hierarchical Forecast Reconciliation
Applies MinT (Minimum Trace) with Weighted Least Squares (WLS) or diagonal covariance to reconcile forecasts, ensuring totals match summed components while minimizing variance.

### 5. Benchmarking Methods
- **ELM Variants**: Base and reconciled (REC) versions of SISO, MIMO, MIMO-MH.
- **External Models**: Prophet (probabilistic univariate) and TimeGPT (pre-trained foundation model).

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

The main script offers a menu-driven interface for selecting models (base, reconciled, or both) and external benchmarks. Hyperparameters (e.g., split date, ELM neurons, regularization) are prompted interactively.

#### Execution Workflow

**1. Interactive Menu System**
   - Displays model options including individual, reconciled, combined, and external groups.
   - Supports custom selections with reconciliation flags.

**2. Automated Data Pipeline**
   - Loads and preprocesses CSV data via `load_and_preprocess_data()`.
   - Splits train/test based on user-defined date.
   - Builds sliding windows for ELM inputs.
   - Runs selected models and stores results.

**3. Execution Options**

| Option | Models Executed | Description |
|--------|-----------------|-------------|
| **1** | SISO | Independent per-source forecasting |
| **2** | MIMO | Joint multi-source per horizon |
| **3** | MIMO-MH | Joint multi-source multi-horizon |
| **4** | SISO-REC | SISO with reconciliation |
| **5** | MIMO-REC | MIMO with reconciliation |
| **6** | MIMO-MH-REC | MIMO-MH with reconciliation |
| **7** | SISO + SISO-REC | Base and reconciled SISO |
| **8** | MIMO + MIMO-REC | Base and reconciled MIMO |
| **9** | MIMO-MH + MIMO-MH-REC | Base and reconciled MIMO-MH |
| **10** | All ELM (Base + REC) | Full ELM suite |
| **11** | Prophet + TimeGPT | External benchmarks |
| **12** | All ELM Base + External | Base ELM + benchmarks |
| **13** | All ELM REC + External | Reconciled ELM + benchmarks |
| **14** | Custom | User-defined combinations |

#### Output Generation

The pipeline automatically generates:
- Forecast results DataFrame
- Metrics CSV (e.g., `final_metrics_comparison.csv`)
- Performance plots (subplots, normalized metrics, by-variable views) saved to `figures/`
- Execution timings and logs

## Key Features

- **Efficiency**: ELM's single-layer, closed-form training enables rapid iterations.
- **Coherence**: Reconciliation enforces physical sums (total = sum of sources).
- **Flexibility**: Interactive menu for model/hyperparameter selection.
- **Comparability**: Integrates external models; computes normalized metrics.
- **Visualization**: Multi-faceted plots for horizon-wise and source-wise analysis.
- **Robustness**: Handles missing data, duplicates; non-negative predictions via ReLU clamping.

## References
**<a id="ref1">[1]</a>** Sheraz Aslam, Herodotos Herodotou, Syed Muhammad Mohsin, Nadeem Javaid, Nouman Ashraf, and Shahzad Aslam. [A survey on deep learning methods for power load and renewable energy forecasting in smart microgrids](https://doi.org/10.1016/j.rser.2021.110992). Renewable and Sustainable Energy Reviews, 144:110992, 2021.

**<a id="ref2">[2]</a>** Gilles Notton, Marie-Laure Nivet, Cyril Voyant, Christophe Paoli, Christophe Darras, Fabrice Motte, and Alexis Fouilloy. [Intermittent and stochastic character of renewable energy sources: Consequences, cost of intermittence and benefit of forecasting](https://doi.org/10.1016/j.rser.2018.02.007). Renewable and Sustainable Energy Reviews, 87:96–105, 2018.

**<a id="ref3">[3]</a>** Cyril Voyant, Milan Despotovic, Gilles Notton, Yves-Marie Saint-Drenan, Mohammed Asloune, and Luis Garcia-Gutierrez. [On the importance of clear-sky model in short-term solar radiation forecasting](https://doi.org/10.1016/j.solener.2025.113490). Solar Energy, 294:113490, 2025.


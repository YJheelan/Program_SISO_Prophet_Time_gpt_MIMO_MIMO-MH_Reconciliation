# ==============================================================================
# elm.py
# ==============================================================================
# Extreme Learning Machine (ELM) model implementation.
from scipy.linalg import pinv
import numpy as np
import pandas as pd
from pathlib import Path

def train_elm(X_train, Y_train, X_test, Y_test,
              num_hidden=1000, num_initializations=1, lambda_reg=1e-6):
    """
    Train an ELM (Extreme Learning Machine) with Ridge-regularized linear readout.

    MODEL
    -----
    • Hidden layer: random affine transform + ReLU activation:
          H = ReLU( X_train @ W^T + b^T )
      where W ∈ R[num_hidden, n_features], b ∈ R[num_hidden, 1].
    • Output/readout layer (beta): closed-form ridge regression per output:
          (H^T H + λ I) β = H^T Y_train
      solved via np.linalg.solve; falls back to pseudo-inverse if needed.

    SEEDS / INITIALIZATIONS
    -----------------------
    • We draw new random W, b for each initialization and keep the one with the
      lowest validation RMSE on X_test/Y_test.
    • The number of tries is `num_initializations` (a.k.a. seeds / restarts).

    PREDICTIONS
    -----------
    • Test hidden: H_test = ReLU( X_test @ W^T + b^T )
    • Predictions:  Y_pred = H_test @ β
      (Your original code applies ReLU to the output as well and clips negatives
       to 0 via np.maximum; we keep that behavior.)

    METRIC
    ------
    • RMSE averaged across output dimensions:
          mean_h( sqrt( mean_t( (y_true - y_pred)^2 ) ) )

    External config expected (if args are None)
    -------------------------------------------
    NUM_INITIALIZATIONS = 1             # number of random initializations (seeds)
    N_HIDDEN = 1000                     # 1000 neurons
    LAMBDA_REG = 1e-6                   # regularization

    Returns
    -------
    best_model : tuple (W, b, beta)
        Parameters for the best run: weights, biases, and readout matrix.
    """
    best_rmse = float('inf')
    best_model = None
    for _ in range(num_initializations):
        # 1. Initialize random input weights (W) and biases (b)
        W = np.random.rand(num_hidden, X_train.shape[1])
        b = np.random.rand(num_hidden, 1)
        
        # 2. Calculate the hidden layer output matrix (H)
        H = np.maximum(0, X_train @ W.T + b.T) # ReLU activation
        
        # 3. Ridge-regularized closed-form solution for beta (output weights).
        HTH = H.T @ H               # (m, m)
        HTY = H.T @ Y_train         # (m, n_outputs or flattened dim)
        try:
            # More stable solution using solve
            beta = np.linalg.solve(HTH + lambda_reg * np.eye(HTH.shape[0]), HTY)
        except np.linalg.LinAlgError:
            # Fallback to Moore-Penrose pseudo-inverse if solve fails
            beta = pinv(H) @ Y_train
            
        # Evaluate this initialization on the test set
        H_test = np.maximum(0, X_test @ W.T + b.T)
        Y_pred = np.maximum(H_test @ beta, 0) # Ensure non-negative predictions
        rmse = np.mean(np.sqrt(np.mean((Y_test - Y_pred) ** 2, axis=0)))
        
        # Keep the best model
        if rmse < best_rmse:
            best_model = (W, b, beta)
            best_rmse = rmse
            
    return best_model

def predict_elm(X, model):
    """
    Forward pass for a trained ELM.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    model : tuple
        (W, b, beta) returned by `train_elm`.

    Returns
    -------
    Y_pred : np.ndarray
        Predicted targets. Uses the same ReLU on hidden and non-negativity clamp
        on outputs as in training selection.
    """
    W, b, beta = model
    H = np.maximum(0, X @ W.T + b.T)
    Y_pred = np.maximum(H @ beta, 0)
    return Y_pred
# ==============================================================================
# reconciliation.py
# ==============================================================================
import numpy as np

# ------------------------------------------------------------
# Helpers for reconciliation (built for your TARGETS ordering) (MINT)
# ------------------------------------------------------------

def create_S_from_outputs(output_names):
    """
    Build the aggregation matrix S given output_names in the order:
    [total, bottom_1, ..., bottom_m].

    Returns S of shape ((m+1) x m) such that y_all = S @ b,
    where b are the m bottom-level series.
    """
    assert len(output_names) >= 2, "Need at least total + one bottom series."
    total_label = output_names[0]
    assert any(k in total_label.lower() for k in ["production_totale", "total", "tot"]), \
        "First output must be the total series."

    m = len(output_names) - 1  # number of bottom series
    S = np.zeros((m + 1, m), dtype=float)
    S[0, :] = 1.0              # total is the sum of all bottoms
    S[1:, :] = np.eye(m)       # each bottom maps to itself
    #print('S',S)
    return S

def estimate_W_from_train_residuals(Y_true_mh_train, Y_pred_mh_train, diagonal=True):
    """
    Estimate error (co)variance per horizon from TRAIN residuals.

    Parameters
    ----------
    Y_true_mh_train : (n_samples, n_horizons, n_outputs)
    Y_pred_mh_train : (n_samples, n_horizons, n_outputs)
    diagonal        : if True, return diagonal W (variances only)

    Returns
    -------
    W_list : list of length n_horizons, each W is (n_outputs x n_outputs)
    """
    n_h = Y_true_mh_train.shape[1]
    n_o = Y_true_mh_train.shape[2]
    W_list = []

    for h in range(n_h):
        E = (Y_true_mh_train[:, h, :] - Y_pred_mh_train[:, h, :])  # (n_samples, n_o)
        if diagonal:
            v = np.var(E, axis=0, ddof=1)
            v = np.maximum(v, 1e-8)  # numerical floor
            W = np.diag(v)
        else:
            # Full covariance + small ridge for stability
            W = np.cov(E.T, ddof=1)
            lam = 1e-6 * np.trace(W) / n_o
            W = W + lam * np.eye(n_o)
        W_list.append(W)

    return W_list

def reconcile_all_horizons(Y_pred_mh, S, W_list):
    """
    MinT-Diagonal (WLS) reconciliation per horizon.

    Parameters
    ----------
    Y_pred_mh : (n_samples, n_horizons, n_outputs) ordered as [total, bottom_1..m]
    S         : ((m+1) x m) aggregation matrix
    W_list    : list of length n_horizons, each W is (n_outputs x n_outputs)

    Returns
    -------
    Y_rec : reconciled forecasts, same shape as Y_pred_mh
    """
    n_s, n_h, n_o = Y_pred_mh.shape
    m = S.shape[1]
    assert n_o == m + 1, "outputs must be [total + m bottom series] to match S."

    Y_rec = np.empty_like(Y_pred_mh)

    for h in range(n_h):
        W = W_list[h]
        # Use pseudo-inverses for numerical robustness
        W_inv = np.linalg.pinv(W)
        StW_inv = S.T @ W_inv
        P = np.linalg.pinv(StW_inv @ S) @ StW_inv      # (m x (m+1))
        H = S @ P                                      # (m+1 x (m+1)) projection

        Yh = Y_pred_mh[:, h, :].T                      # (n_o, n_s)
        Yh_rec = H @ Yh                                # (n_o, n_s)
        Y_rec[:, h, :] = Yh_rec.T

    return Y_rec

# ------------------------------------------------------------
# Helpers for reconciliation (built for your TARGETS ordering) (WLS)
# ------------------------------------------------------------
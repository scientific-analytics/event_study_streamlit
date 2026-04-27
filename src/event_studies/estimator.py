from typing import Tuple
import numpy as np

def clean_estimation_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1) # FIXME this will drop each row even if there is just one return missing across assets
    return X[valid_mask], y[valid_mask]

def clean_estimation_X(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_mask = ~np.isnan(X).any(axis=1)
    return X[valid_mask], y[valid_mask]

def compute_abnormal_returns(X_event: np.ndarray, params: np.ndarray, y_event: np.ndarray, X_est: np.ndarray, y_est: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    expected_returns = X_event @ params
    abnormal_returns = y_event - expected_returns

    y_pred = X_est @ params
    residuals = y_est - y_pred
    valid_residuals = ~np.isnan(residuals)
    dof_per_security = valid_residuals.sum(axis=0) - X_est.shape[1]
    residual_sum_squares = np.nansum(residuals ** 2, axis=0)
    sigma2 = residual_sum_squares / dof_per_security

    # Fixed: XtX_inv and H should be computed per security using only rows where y_est[:, i] is not NaN,
    # to match the observations used in sigma2.
    # Event-window rows with missing factor data (NaN in X_event) produce NaN in
    # the entire corresponding row and column of H.  0 * NaN = NaN in IEEE 754,
    # so even iota[t]=0 doesn't protect the car_variances quadratic form from NaN
    # propagation on all subsequent days.  Zero those rows/cols out so missing
    # event days contribute 0 to the quadratic form rather than poisoning it.
    valid_evt = ~np.isnan(X_event).any(axis=1)

    IplusH = np.empty((len(sigma2), X_event.shape[0], X_event.shape[0]))  # (N, T_event, T_event)
    ar_variances = np.empty((X_event.shape[0], len(sigma2)))               # (T_event, N)
    for i, s2 in enumerate(sigma2):
      X_est_i = X_est[~np.isnan(y_est[:, i])]
      XtX_inv_i = np.linalg.pinv(X_est_i.T @ X_est_i)
      H_raw = X_event @ XtX_inv_i @ X_event.T
      H_raw[~valid_evt, :] = 0.0
      H_raw[:, ~valid_evt] = 0.0
      IplusH[i] = np.eye(X_event.shape[0]) + H_raw
      ar_variances[:, i] = s2 * np.diag(IplusH[i])
    #
    # # Note: IplusH becomes (N, T_event, T_event) — update the downstream TODO accordingly
    # XtX_inv = np.linalg.pinv(X_est.T @ X_est)
    # H = X_event @ XtX_inv @ X_event.T
    # variance_multiplier = np.diag(np.eye(H.shape[0]) + H) #FIXME: np.diag overlooks the covariances over the different days
    # ar_variances = np.array([s2 * variance_multiplier for s2 in sigma2]).T

    return abnormal_returns, ar_variances, IplusH, sigma2

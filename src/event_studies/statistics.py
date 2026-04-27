from typing import Tuple
import numpy as np
from scipy import stats

def significance_test_ar(abnormal_returns: np.ndarray, ar_variances: np.ndarray, degrees_of_freedom: int, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(divide='ignore', invalid='ignore'):
        t_stats = abnormal_returns / np.sqrt(ar_variances)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=degrees_of_freedom))
    significant = p_values < alpha
    return t_stats, p_values, significant


def significance_test_car(abnormal_returns: np.ndarray, ar_variances: np.ndarray, degrees_of_freedom: int, alpha: float = 0.05,
                          IplusH: np.ndarray = None, sigma2: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    car = np.nancumsum(abnormal_returns, axis=0)
    T, N = abnormal_returns.shape
    if IplusH is not None and sigma2 is not None:
        # Var(CAR_t, i) = sigma2[i] * iota_t' @ IplusH[i][:t+1, :t+1] @ iota_t
        # iota_t has 1s for valid (non-NaN) days up to t, 0s for missing days
        car_variances = np.empty((T, N))
        for i in range(N):
            for t in range(T):
                valid = ~np.isnan(abnormal_returns[:t + 1, i])
                if valid.any():
                    iota = valid.astype(float)
                    car_variances[t, i] = sigma2[i] * iota @ IplusH[i][:t + 1, :t + 1] @ iota
                else:
                    car_variances[t, i] = np.nan
    else:
        car_variances = np.nancumsum(ar_variances, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        car_t_stats = car / np.sqrt(car_variances)
    car_p_values = 2 * (1 - stats.t.cdf(np.abs(car_t_stats), df=degrees_of_freedom))
    car_significant = car_p_values < alpha
    return car, car_variances, car_t_stats, car_p_values, car_significant



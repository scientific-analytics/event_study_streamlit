from typing import Optional
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class EventStudyModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self._is_fitted = False
        self._parameters: Optional[np.ndarray] = None

    @abstractmethod
    def create_design_matrix(self, factors: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_parameter_names(self) -> list[str]:
        pass

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def parameters(self) -> Optional[np.ndarray]:
        return self._parameters

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._parameters = np.linalg.lstsq(X, y, rcond=None)[0]
        self._is_fitted = True

    def fit_single(self, X: np.ndarray, y: np.ndarray, rcond=None) -> None:
        """
        Multi-output OLS with per-target missing='drop' behavior:
        - drop rows where X has any NaN (shared across all targets)
        - for each target column j, also drop rows where y[:, j] is NaN
        - fit one target at a time using np.linalg.lstsq
        Stores parameters with shape:
          - (n_features,) if y is 1D
          - (n_features, n_targets) if y is 2D
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features). Got shape {X.shape}.")

        n, p = X.shape

        # Normalize y to 2D for fitting
        y_was_1d = (y.ndim == 1)
        if y_was_1d:
            y2 = y.reshape(-1, 1)
        elif y.ndim == 2:
            y2 = y
        else:
            raise ValueError(f"y must be 1D or 2D. Got shape {y.shape}.")

        if y2.shape[0] != n:
            raise ValueError(f"X and y must have same number of rows. Got {n} and {y2.shape[0]}.")

        # Rows valid for X (common across all targets)
        x_ok = ~np.isnan(X).any(axis=1)

        # Allocate parameter matrix
        k = y2.shape[1]
        B = np.full((p, k), np.nan, dtype=float)

        for j in range(k):
            mask = x_ok & ~np.isnan(y2[:, j])

            # Not enough rows -> leave NaNs (or raise if you'd rather)
            if mask.sum() == 0:
                continue

            # Solve for this target
            B[:, j] = np.linalg.lstsq(X[mask], y2[mask, j], rcond=rcond)[0]

        # Store parameters in a shape consistent with input y
        self._parameters = B[:, 0] if y_was_1d else B
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self._parameters

    def get_degrees_of_freedom(self, n_observations: int) -> int:
        return n_observations - len(self.get_parameter_names())

class ConstantMeanModel(EventStudyModel):
    def __init__(self):
        super().__init__("constant_model")

    def create_design_matrix(self, factors: pd.DataFrame) -> np.ndarray:
        return np.ones((len(factors), 1))

    def get_parameter_names(self) -> list[str]:
        return ["alpha"]

class MarketModel(EventStudyModel):
    def __init__(self, market_factor: str = "Mkt"):
        super().__init__("market_model")
        self.market_factor = market_factor

    def create_design_matrix(self, factors: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(factors)), factors[self.market_factor].values])

    def get_parameter_names(self) -> list[str]:
        return ["alpha", f"beta_{self.market_factor}"]


class ThreeFactorModel(EventStudyModel):
    def __init__(self, market_factor: str = "Mkt", size_factor: str = "SMB", value_factor: str = "HML"):
        super().__init__("three_factor_model")
        self.market_factor = market_factor
        self.size_factor = size_factor
        self.value_factor = value_factor

    def create_design_matrix(self, factors: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            np.ones(len(factors)),
            factors[self.market_factor].values,
            factors[self.size_factor].values,
            factors[self.value_factor].values
        ])

    def get_parameter_names(self) -> list[str]:
        return ["alpha", f"beta_{self.market_factor}", f"beta_{self.size_factor}", f"beta_{self.value_factor}"]

class FourFactorModel(EventStudyModel):
    def __init__(self, market_factor: str = "Mkt", size_factor: str = "SMB", value_factor: str = "HML", momentum_factor: str = "UMD"):
        super().__init__("four_factor_model")
        self.market_factor = market_factor
        self.size_factor = size_factor
        self.value_factor = value_factor
        self.momentum_factor = momentum_factor

    def create_design_matrix(self, factors: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            np.ones(len(factors)),
            factors[self.market_factor].values,
            factors[self.size_factor].values,
            factors[self.value_factor].values,
            factors[self.momentum_factor].values
        ])

    def get_parameter_names(self) -> list[str]:
        return ["alpha", f"beta_{self.market_factor}", f"beta_{self.size_factor}", f"beta_{self.value_factor}", f"beta_{self.momentum_factor}"]

def get_model(model_name: str, **kwargs) -> EventStudyModel:
    model_map = {
        'constant_model': ConstantMeanModel,
        'market_model': MarketModel,
        'three_factor_model': ThreeFactorModel,
        'four_factor_model': FourFactorModel,
    }
    return model_map[model_name](**kwargs)
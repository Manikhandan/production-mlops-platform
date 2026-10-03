from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from mlops_platform.exceptions import FeatureSchemaError
from mlops_platform.schemas import N_FEATURES


class FeaturePipeline:
    """Scale features the same way at train and serve time."""

    def __init__(self, scaler: StandardScaler | None = None) -> None:
        self.scaler = scaler or StandardScaler()
        self._fitted = scaler is not None

    def fit(self, X: np.ndarray) -> FeaturePipeline:
        if X.shape[1] != N_FEATURES:
            raise FeatureSchemaError(f"expected {N_FEATURES} columns")
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise FeatureSchemaError("feature pipeline is not fitted")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != N_FEATURES:
            raise FeatureSchemaError(f"expected {N_FEATURES} columns, got {X.shape[1]}")
        return self.scaler.transform(X)

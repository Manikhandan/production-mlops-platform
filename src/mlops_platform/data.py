from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlops_platform.exceptions import DataValidationError
from mlops_platform.schemas import FEATURE_NAMES, N_FEATURES


@dataclass(frozen=True)
class Dataset:
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    feature_names: tuple[str, ...]


def generate_dataset(n_samples: int = 2400, seed: int = 7) -> Dataset:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=2,
        class_sep=1.15,
        flip_y=0.02,
        random_state=seed,
    )
    validate_matrix(X, y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    return Dataset(X_train, X_valid, y_train, y_valid, FEATURE_NAMES)


def validate_matrix(X: np.ndarray, y: np.ndarray | None = None) -> None:
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise DataValidationError(f"expected shape (n, {N_FEATURES}), got {X.shape}")
    if not np.isfinite(X).all():
        raise DataValidationError("feature matrix contains non-finite values")
    if y is not None:
        if y.shape[0] != X.shape[0]:
            raise DataValidationError("label length does not match rows")
        if set(np.unique(y)) - {0, 1}:
            raise DataValidationError("labels must be binary 0/1")


def persist_dataset(dataset: Dataset, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "split.npz",
        X_train=dataset.X_train,
        X_valid=dataset.X_valid,
        y_train=dataset.y_train,
        y_valid=dataset.y_valid,
        feature_names=np.array(dataset.feature_names),
    )

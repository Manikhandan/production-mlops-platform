from __future__ import annotations

import math

import numpy as np


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between two 1-d distributions. 0.1 watch, 0.2 investigate."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) < 20 or len(current) < 20:
        return 0.0
    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_p = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-4, 1)
    cur_p = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-4, 1)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def multivariate_psi(reference: np.ndarray, current: np.ndarray) -> dict[str, float]:
    if reference.ndim != 2 or current.ndim != 2:
        raise ValueError("expected 2-d arrays")
    per_feature = [
        population_stability_index(reference[:, i], current[:, i])
        for i in range(reference.shape[1])
    ]
    mean = float(np.mean(per_feature)) if per_feature else 0.0
    return {
        "psi_mean": mean,
        "psi_max": float(max(per_feature) if per_feature else 0.0),
        "n_current": float(len(current)),
    }


def drifted(psi_mean: float, threshold: float) -> bool:
    return psi_mean >= threshold


def finite_or_nan(value: float) -> float:
    return value if math.isfinite(value) else float("nan")

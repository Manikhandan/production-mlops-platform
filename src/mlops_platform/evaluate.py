from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    scores = model.predict_proba(X)[:, 1]
    labels = (scores >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "accuracy": float(accuracy_score(y, labels)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "brier": float(brier_score_loss(y, scores)),
        "n_valid": float(len(y)),
    }

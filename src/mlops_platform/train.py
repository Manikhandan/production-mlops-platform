from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_platform.data import Dataset
from mlops_platform.evaluate import evaluate
from mlops_platform.exceptions import QualityGateError


@dataclass(frozen=True)
class TrainedModel:
    pipeline: Pipeline
    metrics: dict[str, float]
    algorithm: str = "logreg"


def build_estimator(seed: int = 7) -> Pipeline:
    clf = LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)
    return Pipeline([("features", StandardScaler()), ("clf", clf)])


def train(dataset: Dataset, min_roc_auc: float, seed: int = 7) -> TrainedModel:
    pipeline = build_estimator(seed)
    pipeline.fit(dataset.X_train, dataset.y_train)
    metrics = evaluate(pipeline, dataset.X_valid, dataset.y_valid)
    if metrics["roc_auc"] < min_roc_auc:
        raise QualityGateError(
            f"roc_auc {metrics['roc_auc']:.3f} below gate {min_roc_auc:.3f}"
        )
    return TrainedModel(pipeline=pipeline, metrics=metrics)

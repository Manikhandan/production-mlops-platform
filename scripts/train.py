from __future__ import annotations

from pathlib import Path

import numpy as np

from mlops_platform.config import get_settings
from mlops_platform.data import generate_dataset, persist_dataset
from mlops_platform.registry import ModelRegistry
from mlops_platform.train import train


def run(seed: int | None = None) -> dict[str, object]:
    settings = get_settings()
    dataset = generate_dataset(seed=seed or settings.random_seed)
    persist_dataset(dataset, settings.data_dir)
    trained = train(dataset, min_roc_auc=settings.min_roc_auc, seed=settings.random_seed)
    registry = ModelRegistry(settings.registry_dir)
    record = registry.register(trained)
    np.save(Path(record.path).parent / "reference.npy", dataset.X_train)
    staging = registry.promote(record.version, "Staging")
    production = registry.promote(record.version, "Production")
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    return {
        "version": production.version,
        "stage": production.stage,
        "metrics": trained.metrics,
        "prior_stage": staging.stage,
    }


if __name__ == "__main__":
    print(run())

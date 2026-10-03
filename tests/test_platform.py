from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops_platform.config import Settings
from mlops_platform.data import generate_dataset, validate_matrix
from mlops_platform.drift import multivariate_psi, population_stability_index
from mlops_platform.exceptions import DataValidationError, QualityGateError, RegistryError
from mlops_platform.registry import ModelRegistry
from mlops_platform.serving import create_app
from mlops_platform.train import train


@pytest.fixture()
def trained_env(tmp_path: Path):
    settings = Settings(
        registry_dir=tmp_path / "registry",
        artifact_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        request_log_path=tmp_path / "var" / "predictions.jsonl",
        min_roc_auc=0.8,
    )
    dataset = generate_dataset(n_samples=800, seed=7)
    trained = train(dataset, min_roc_auc=settings.min_roc_auc)
    registry = ModelRegistry(settings.registry_dir)
    record = registry.register(trained)
    np.save(Path(record.path).parent / "reference.npy", dataset.X_train)
    registry.promote(record.version, "Production")
    return settings, dataset, record


def test_validate_rejects_wrong_shape():
    with pytest.raises(DataValidationError):
        validate_matrix(np.zeros((4, 2)))


def test_quality_gate_rejects_impossible_auc(tmp_path: Path):
    dataset = generate_dataset(n_samples=400, seed=3)
    with pytest.raises(QualityGateError):
        train(dataset, min_roc_auc=1.01)


def test_registry_promotion_archives_previous(tmp_path: Path):
    dataset = generate_dataset(n_samples=600, seed=1)
    trained = train(dataset, min_roc_auc=0.8)
    registry = ModelRegistry(tmp_path)
    first = registry.register(trained, version="v001")
    second = registry.register(trained, version="v002")
    registry.promote(first.version, "Production")
    registry.promote(second.version, "Production")
    models = {item.version: item.stage for item in registry.list_models()}
    assert models["v001"] == "Archived"
    assert models["v002"] == "Production"
    assert registry.resolve("Production").version == "v002"


def test_registry_rejects_duplicate_version(tmp_path: Path):
    dataset = generate_dataset(n_samples=400, seed=2)
    trained = train(dataset, min_roc_auc=0.8)
    registry = ModelRegistry(tmp_path)
    registry.register(trained, version="v001")
    with pytest.raises(RegistryError):
        registry.register(trained, version="v001")


def test_predict_and_health(trained_env):
    settings, dataset, record = trained_env
    app = create_app(settings)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready").json()
    assert ready["ready"] is True
    assert ready["model_version"] == record.version
    payload = {"features": dataset.X_valid[0].tolist()}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == record.version
    assert 0.0 <= body["score"] <= 1.0
    assert response.headers["x-request-id"]
    metrics = client.get("/metrics")
    assert "ml_requests_total" in metrics.text


def test_invalid_payload_rejected(trained_env):
    settings, _, _ = trained_env
    client = TestClient(create_app(settings))
    bad = client.post("/predict", json={"features": [1, 2]})
    assert bad.status_code == 422
    extra = client.post("/predict", json={"features": [1, 2, 3, 4, 5, 6]})
    assert extra.status_code == 422
    not_number = client.post("/predict", json={"features": ["a", "b", "c", "d", "e"]})
    assert not_number.status_code == 422


def test_ready_false_without_model(tmp_path: Path):
    settings = Settings(registry_dir=tmp_path / "empty")
    client = TestClient(create_app(settings))
    body = client.get("/ready").json()
    assert body["ready"] is False


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, size=400)
    shifted = rng.normal(2.5, 1, size=400)
    assert population_stability_index(reference, shifted) > 0.2
    matrix_ref = rng.normal(0, 1, size=(200, 5))
    matrix_cur = rng.normal(3, 1, size=(200, 5))
    report = multivariate_psi(matrix_ref, matrix_cur)
    assert report["psi_mean"] > 0.2


def test_invalid_promotion_and_missing_alias(tmp_path: Path):
    dataset = generate_dataset(n_samples=400, seed=4)
    trained = train(dataset, min_roc_auc=0.8)
    registry = ModelRegistry(tmp_path / "reg")
    registry.register(trained, version="v001")
    with pytest.raises(RegistryError):
        registry.promote("v001", "Canary")
    with pytest.raises(RegistryError):
        registry.resolve("Production")


def test_health_vs_ready_and_predict_503(tmp_path: Path):
    settings = Settings(registry_dir=tmp_path / "empty")
    client = TestClient(create_app(settings))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["ready"] is False
    denied = client.post("/predict", json={"features": [0.1, 0.2, 0.3, 0.4, 0.5]})
    assert denied.status_code == 503


def test_rollback_by_repromoting(tmp_path: Path):
    dataset = generate_dataset(n_samples=500, seed=5)
    trained = train(dataset, min_roc_auc=0.8)
    registry = ModelRegistry(tmp_path)
    registry.register(trained, version="v001")
    registry.register(trained, version="v002")
    registry.promote("v001", "Production")
    registry.promote("v002", "Production")
    rolled = registry.promote("v001", "Production")
    assert rolled.version == "v001"
    assert registry.resolve("Production").version == "v001"


def test_missing_version_cannot_promote(tmp_path: Path):
    registry = ModelRegistry(tmp_path)
    with pytest.raises(RegistryError):
        registry.promote("v999", "Production")


def test_drift_insufficient_window(trained_env):
    settings, _, _ = trained_env
    client = TestClient(create_app(settings))
    report = client.get("/v1/drift").json()
    assert report["ready"] is False
    assert report["reason"] == "insufficient_window"

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from mlops_platform import metrics
from mlops_platform.config import Settings, get_settings
from mlops_platform.drift import multivariate_psi
from mlops_platform.exceptions import ModelNotReadyError, PlatformError, RegistryError
from mlops_platform.logging import configure_logging
from mlops_platform.registry import ModelRegistry
from mlops_platform.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

log = logging.getLogger("mlops_platform")


class ModelRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = ModelRegistry(settings.registry_dir)
        self.pipeline = None
        self.record = None
        self.reference: np.ndarray | None = None
        self.recent: deque[list[float]] = deque(maxlen=settings.drift_window)
        self._lock = threading.Lock()
        self.settings.request_log_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        pipeline, record = self.registry.load_pipeline(self.settings.stage)
        ref_path = Path(record.path).parent / "reference.npy"
        reference = np.load(ref_path) if ref_path.exists() else None
        with self._lock:
            self.pipeline = pipeline
            self.record = record
            self.reference = reference
        log.info("model_loaded", extra={"model_version": record.version, "stage": record.stage})

    @property
    def ready(self) -> bool:
        return self.pipeline is not None and self.record is not None

    def predict(self, features: list[float], request_id: str) -> PredictResponse:
        if not self.ready or self.pipeline is None or self.record is None:
            raise ModelNotReadyError("no Production model is loaded")
        X = np.asarray(features, dtype=float).reshape(1, -1)
        started = time.perf_counter()
        score = float(self.pipeline.predict_proba(X)[0, 1])
        metrics.LATENCY.observe(time.perf_counter() - started)
        metrics.SCORE.observe(score)
        self.recent.append(features)
        self._append_log(request_id, features, score)
        return PredictResponse(
            label=int(score >= 0.5),
            score=round(score, 6),
            model_version=self.record.version,
            stage=self.record.stage,
            request_id=request_id,
        )

    def drift_report(self) -> dict[str, float | bool | int]:
        if self.reference is None or len(self.recent) < 20:
            return {"ready": False, "reason": "insufficient_window", "n": len(self.recent)}
        current = np.asarray(self.recent, dtype=float)
        stats = multivariate_psi(self.reference, current)
        stats["alert"] = stats["psi_mean"] >= self.settings.drift_alert_psi
        stats["threshold"] = self.settings.drift_alert_psi
        stats["ready"] = True
        return stats

    def _append_log(self, request_id: str, features: list[float], score: float) -> None:
        line = json.dumps(
            {
                "request_id": request_id,
                "features": features,
                "score": score,
                "model_version": None if self.record is None else self.record.version,
            }
        )
        with self.settings.request_log_path.open("a") as handle:
            handle.write(line + "\n")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    runtime = ModelRuntime(settings)
    try:
        runtime.load()
    except RegistryError:
        log.warning("startup_without_model")

    app = FastAPI(title=settings.app_name, version="1.0.0")
    app.state.runtime = runtime
    app.state.settings = settings

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(PlatformError)
    async def platform_error(_, exc: PlatformError):
        metrics.ERRORS.labels(kind=type(exc).__name__).inc()
        status = 503 if isinstance(exc, ModelNotReadyError) else 400
        return JSONResponse({"error": str(exc), "type": type(exc).__name__}, status_code=status)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        if not runtime.ready:
            return ReadyResponse(ready=False, reason="model_not_loaded")
        assert runtime.record is not None
        return ReadyResponse(
            ready=True, model_version=runtime.record.version, stage=runtime.record.stage
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        metrics.REQUESTS.labels(endpoint="predict", status="ok").inc()
        try:
            return runtime.predict(body.features, request.state.request_id)
        except Exception:
            metrics.REQUESTS.labels(endpoint="predict", status="error").inc()
            raise

    @app.get("/v1/models")
    def models() -> dict:
        return {"models": [item.model_dump() for item in runtime.registry.list_models()]}

    @app.get("/v1/drift")
    def drift() -> dict:
        return runtime.drift_report()

    @app.get("/metrics")
    def prometheus_metrics():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": settings.app_name, "docs": "/docs"}

    @app.post("/v1/reload")
    def reload_model() -> dict[str, str]:
        try:
            runtime.load()
        except RegistryError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        assert runtime.record is not None
        return {"reloaded": runtime.record.version}

    return app


app = create_app()

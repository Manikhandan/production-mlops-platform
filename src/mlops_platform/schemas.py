from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

FEATURE_NAMES = ("amount", "velocity", "account_age_days", "channel", "region")
N_FEATURES = len(FEATURE_NAMES)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: list[float] = Field(..., min_length=N_FEATURES, max_length=N_FEATURES)

    @field_validator("features")
    @classmethod
    def finite(cls, value: list[float]) -> list[float]:
        if any(v != v or v in (float("inf"), float("-inf")) for v in value):
            raise ValueError("features must be finite numbers")
        return value


class PredictResponse(BaseModel):
    label: int
    score: float
    model_version: str
    stage: str
    request_id: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_version: str | None = None
    stage: str | None = None
    reason: str | None = None


class PromoteRequest(BaseModel):
    version: str
    stage: str = "Production"


class RegistryRecord(BaseModel):
    version: str
    stage: str
    path: str
    metrics: dict[str, Any]
    created_at: str
    algorithm: str

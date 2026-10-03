from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Secrets come from the environment, never from code."""

    model_config = SettingsConfigDict(
        env_prefix="MLOPS_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "mlops-platform"
    log_level: str = "INFO"
    registry_dir: Path = Field(default=Path("models/registry"))
    artifact_dir: Path = Field(default=Path("models/artifacts"))
    data_dir: Path = Field(default=Path("data"))
    stage: str = "Production"
    request_log_path: Path = Field(default=Path("var/predictions.jsonl"))
    drift_window: int = 256
    drift_alert_psi: float = 0.2
    metrics_enabled: bool = True
    min_roc_auc: float = 0.85
    random_seed: int = 7


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

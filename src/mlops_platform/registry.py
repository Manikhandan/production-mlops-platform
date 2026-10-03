from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from mlops_platform.exceptions import RegistryError
from mlops_platform.schemas import RegistryRecord
from mlops_platform.train import TrainedModel

STAGES = ("None", "Staging", "Production", "Archived")


class ModelRegistry:
    """File-backed registry. Promotion is an explicit write, not an overwrite of artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "index.json"
        if not self._index.exists():
            self._write_index({"models": [], "aliases": {}})

    def _read_index(self) -> dict[str, Any]:
        return json.loads(self._index.read_text())

    def _write_index(self, payload: dict[str, Any]) -> None:
        tmp = self._index.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self._index)

    def register(self, trained: TrainedModel, version: str | None = None) -> RegistryRecord:
        index = self._read_index()
        version = version or f"v{len(index['models']) + 1:03d}"
        if any(item["version"] == version for item in index["models"]):
            raise RegistryError(f"version already exists: {version}")
        dest = self.root / version
        dest.mkdir(parents=True, exist_ok=False)
        artifact = dest / "model.joblib"
        joblib.dump(trained.pipeline, artifact)
        record = RegistryRecord(
            version=version,
            stage="None",
            path=str(artifact),
            metrics=trained.metrics,
            created_at=datetime.now(timezone.utc).isoformat(),
            algorithm=trained.algorithm,
        )
        meta = dest / "metadata.json"
        meta.write_text(record.model_dump_json(indent=2))
        index["models"].append(record.model_dump())
        self._write_index(index)
        return record

    def promote(self, version: str, stage: str) -> RegistryRecord:
        if stage not in STAGES:
            raise RegistryError(f"unknown stage {stage}")
        index = self._read_index()
        found = None
        for item in index["models"]:
            if item["version"] == version:
                found = item
            elif item["stage"] == stage and stage in {"Staging", "Production"}:
                item["stage"] = "Archived"
        if found is None:
            raise RegistryError(f"unknown version {version}")
        found["stage"] = stage
        index["aliases"][stage] = version
        Path(found["path"]).parent.joinpath("metadata.json").write_text(
            json.dumps(found, indent=2)
        )
        self._write_index(index)
        return RegistryRecord.model_validate(found)

    def resolve(self, stage: str) -> RegistryRecord:
        index = self._read_index()
        version = index["aliases"].get(stage)
        if not version:
            raise RegistryError(f"no model promoted to {stage}")
        for item in index["models"]:
            if item["version"] == version:
                return RegistryRecord.model_validate(item)
        raise RegistryError(f"alias {stage} points at missing version {version}")

    def list_models(self) -> list[RegistryRecord]:
        return [RegistryRecord.model_validate(item) for item in self._read_index()["models"]]

    def load_pipeline(self, stage: str):
        record = self.resolve(stage)
        return joblib.load(record.path), record

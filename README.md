# Production ML Deployment Platform

An ML engineering and serving path for a tabular binary classifier: data validation, feature scaling, training, holdout evaluation, a quality gate, an immutable registry, Staging → Production promotion, FastAPI inference, and a PSI window on live features.

This is original engineering work in a public repository. It is not a cloud deployment and does not use live traffic.

## Engineering problem

A fitted sklearn object is not a production path. The gap is everything around it: schema, validation, a registry that does not overwrite artifacts, an explicit promotion step, a process that can refuse traffic when the model is missing, and a way to notice that incoming features no longer look like the training reference.

## Architecture

```
Client
  │
  ▼
FastAPI ── /health  (liveness)
         ── /ready   (model loaded + stage)
         ── /predict (schema → score → jsonl)
         ── /metrics (Prometheus)
         ── /v1/drift (PSI vs training reference)
         ── /v1/models
                │
                ▼
        ModelRuntime (in-process)
                │
                ▼
        File registry (models/registry)
          ├── v00N/model.joblib
          ├── v00N/metadata.json
          ├── v00N/reference.npy
          └── index.json  (aliases: Staging, Production)
                ▲
                │
Train job ── quality gate (roc_auc) ── register ── promote
```

Kubernetes manifests under `infra/k8s/` describe a rolling Deployment, HPA, PDB, and a separate canary Deployment that loads the Staging alias. They are infrastructure definitions, not evidence that a cluster is running.

## Why this shape

- **File registry instead of MLflow as a hard dependency.** The promotion contract is the thing to get right. MLflow can wrap the same record later (`pip install -e ".[mlflow]"`) without changing serving.
- **Staging then Production.** Production is an alias, not a mutated file. Rollback is `scripts/promote.py v00N --stage Production` plus `POST /v1/reload`.
- **Readiness is not liveness.** Kubernetes can restart a wedged process without taking traffic on a replica that has no model.
- **Drift is sampled from live requests**, compared to the training matrix stored next to the artifact. PSI is a diagnostic, not an automatic retrain.

## Repository structure

```
src/mlops_platform/   application code
scripts/              train + promote entrypoints
tests/                API, registry, gate, drift
configs/              promotion policy
infra/k8s/            rolling + canary manifests
prometheus/ grafana/  scrape config, example dashboard
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python scripts/train.py
uvicorn mlops_platform.serving:app --reload --port 8000
```

```bash
curl -s localhost:8000/ready
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"features":[0.1,0.2,0.3,0.4,0.5]}'
```

## Tests

```bash
ruff check src tests scripts
pytest -q
```

## Deployment

- Docker: `docker build -t mlops-platform:local .`
- Compose: `docker compose up --build` trains once, then serves.
- Kubernetes: apply `infra/k8s/serving.yaml`. Point the image at a registry you control. Model artifacts need a volume the train job writes and the API reads.
- Rollback: promote the previous version; call `/v1/reload` or roll the Deployment.

## Observability

`/metrics` exports request count, errors, latency, and score histogram. `/v1/drift` returns PSI over the recent window. Alert rules in `prometheus/alerts.yml` are starting points, not SLOs.

## Security

- No credentials in the repo. `.env` is gitignored; `.env.example` is the contract.
- Container runs as uid 10001, non-root, with dropped capabilities in the manifest.
- Predict payloads are Pydantic-validated; non-finite values are rejected.
- Image reference in manifests is `replace-me` on purpose.

## Failure handling

| Failure | Behaviour |
| --- | --- |
| No Production alias | `/ready` is false; `/predict` returns 503 |
| roc_auc below gate | `QualityGateError`, nothing registered |
| Duplicate version | `RegistryError` |
| Invalid payload | HTTP 422 |
| Feature shift | `/v1/drift` `alert=true` when PSI ≥ threshold |

## Trade-offs

- In-process model load is enough for one replica-local artifact. A remote store (S3/GCS) is the next step at more than one writer.
- Logistic regression on synthetic data keeps CI fast and deterministic. Swap the estimator in `train.py` without changing the registry contract.
- PSI on a sliding window will false-alert on tiny samples; the endpoint stays silent until 20 observations.

## What I would improve next

- Export PSI as a Prometheus gauge from a sidecar or periodic job.
- Sign artifacts and verify digest at load.
- Shadow traffic against Staging before flipping the Production alias.

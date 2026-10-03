# Architecture

## Control vs data plane

The **training job** is a batch control-plane action: generate (or read) a split, fit, evaluate, register, promote. The **API** is the data plane: validated requests, a loaded pipeline, logs, metrics.

They share the registry directory and nothing else. That split is what lets you schedule training on a Job while serving stays on a Deployment.

## Sequence

1. `scripts/train.py` writes `data/split.npz`, fits, evaluates.
2. If `roc_auc >= min_roc_auc`, a new `v00N/` directory is created. Artifacts are append-only.
3. The job promotes to Staging, then Production. Previous Production becomes Archived.
4. API process loads the Production alias at start (and on `/v1/reload`).
5. Each `/predict` appends jsonl and records Prometheus metrics.
6. `/v1/drift` compares the recent window to `reference.npy`.

## Rollback

Promotion is the rollback mechanism. Point Production at `v00N-1`, reload. Do not delete the old directory.

## Canary

`infra/k8s/canary.yaml` runs one replica with `MLOPS_STAGE=Staging`. Keep it off the default Service until you want a percentage of traffic. This repository does not implement a service mesh; the manifest is the contract.

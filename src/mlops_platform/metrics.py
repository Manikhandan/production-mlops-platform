from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

registry = CollectorRegistry()

REQUESTS = Counter(
    "ml_requests_total",
    "Inference requests",
    ["endpoint", "status"],
    registry=registry,
)
ERRORS = Counter(
    "ml_errors_total",
    "Handled inference errors",
    ["kind"],
    registry=registry,
)
LATENCY = Histogram(
    "ml_inference_latency_seconds",
    "Scoring latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry,
)
SCORE = Histogram(
    "ml_prediction_score",
    "Predicted positive-class score",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry,
)


def render() -> bytes:
    return generate_latest(registry)

# Runbook: model promotion

Production aliases point at a registered version. Promote Staging only after a holdout eval.

# Incident: replica not ready

If `/ready` is false, the registry has no Production alias. Do not increase replicas until a model is promoted.

# Incident: drift alert

Read `/v1/drift`. If psi_mean exceeds the threshold, freeze promotions and rerun training on a recent window.

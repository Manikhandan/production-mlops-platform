# Inference retry policy

Training jobs are idempotent per `run_id`. Re-running register with a new version never mutates an older artifact.

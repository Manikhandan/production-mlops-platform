# Rollback checklist

1. Identify last known-good version from `GET /v1/models`.
2. `python scripts/promote.py v00N --stage Production`
3. `POST /v1/reload` or roll the Deployment.
4. Confirm `/ready` shows that version.

# Local development

```
pip install -e ".[dev]"
python scripts/train.py
uvicorn mlops_platform.serving:app --port 8000
```

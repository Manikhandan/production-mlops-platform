FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app --uid 10001 app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs
RUN pip install --no-cache-dir . && mkdir -p /app/models /app/var /app/data \
    && chown -R app:app /app

USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "mlops_platform.serving:app", "--host", "0.0.0.0", "--port", "8000"]

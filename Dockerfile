# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MRISK_HOST=0.0.0.0 \
    MRISK_PORT=8000

WORKDIR /app

# Install the package first (better layer caching).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Application assets and the trained artifact.
COPY static ./static
COPY models ./models

EXPOSE 8000

# Fail fast if the model artifact is missing at container start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health').status==200 else 1)"

CMD ["python", "-m", "uvicorn", "maternal_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]

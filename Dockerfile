FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SAFARI_DOCKER=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY scripts ./scripts
COPY docs ./docs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8765

CMD ["python", "scripts/serve_safari.py", "--host", "0.0.0.0", "--port", "8765", "--data-dir", "/data", "--output-dir", "/outputs", "--model", "/models/safari_lgbm_v0.txt"]

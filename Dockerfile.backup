FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    HF_HOME=/app/models \
    HF_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english \
    HF_DEVICE=cpu

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
  && rm -rf /var/lib/apt/lists/*

# Dependencies first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download transformer model weights when available (best effort)
RUN python - <<'PY'
import os
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
except Exception as exc:
    print(f'Skipping transformer cache warmup: {exc}')
else:
    model_name = os.environ.get('HF_MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')
    _ = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModelForSequenceClassification.from_pretrained(model_name)
    _ = AutoModel.from_pretrained(model_name)
    print('Models cached to', os.environ.get('HF_HOME', '<unknown>'))
PY

# Copy application source
COPY . /app

# Create non-root user and set permissions on /app and /app/models
RUN useradd -u 1001 -m appuser \
 && mkdir -p /app/models \
 && chown -R appuser:appuser /app /app/models

USER appuser

EXPOSE 8000

# Start with Gunicorn + UvicornWorker (4 workers)
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "model_server:app", "-b", "0.0.0.0:8000"]

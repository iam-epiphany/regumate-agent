FROM node:22-bookworm-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.13-slim AS app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    REGUMATE_MODEL_CACHE_DIR=/app/data/model_cache \
    HF_HOME=/app/data/model_cache/huggingface \
    HF_HUB_CACHE=/app/data/model_cache/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/app/data/model_cache/sentence_transformers \
    TORCH_HOME=/app/data/model_cache/torch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY data/regulations/ ./data/regulations/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

RUN mkdir -p /app/data/documents/originals /app/data/qdrant /app/data/model_cache

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

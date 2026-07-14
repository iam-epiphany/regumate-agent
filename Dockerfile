FROM node:22-bookworm-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.13-bookworm AS app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    REGUMATE_MODEL_CACHE_DIR=/app/data/model_cache \
    REGUMATE_OFFLINE_MODE=true \
    HF_HOME=/app/data/model_cache/huggingface \
    HF_HUB_CACHE=/app/data/model_cache/huggingface/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/data/model_cache/sentence_transformers \
    TORCH_HOME=/app/data/model_cache/torch \
    EMBEDDING_MODEL_PATH=/app/data/models/bge-m3 \
    RERANKER_MODEL_PATH=/app/data/models/bge-reranker-v2-m3

WORKDIR /app

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends --fix-missing \
        antiword \
        build-essential \
        curl \
        fonts-noto-cjk \
        libreoffice-calc \
        libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-cuda.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-cuda.txt \
    && grep -v '^torch==' requirements.txt > /tmp/requirements-no-torch.txt \
    && python -m pip install -r /tmp/requirements-no-torch.txt

COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY data/regulations/ ./data/regulations/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

RUN mkdir -p /app/data/documents/originals /app/data/qdrant /app/data/model_cache

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

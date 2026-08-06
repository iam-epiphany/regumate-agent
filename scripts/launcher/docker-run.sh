#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/../.."

env_file_value() {
  name="$1"
  if [ -f .env ]; then
    sed -n "s/^${name}=//p" .env | head -n 1 | sed "s/^['\"]//;s/['\"]$//"
  fi
}

env_or_default() {
  name="$1"
  default="$2"
  eval "value=\${$name:-}"
  if [ -n "$value" ]; then
    printf '%s' "$value"
    return
  fi
  file_value="$(env_file_value "$name" || true)"
  if [ -n "$file_value" ]; then
    printf '%s' "$file_value"
    return
  fi
  printf '%s' "$default"
}

default_torch_flavor() {
  if [ "$(uname -s)" = "Darwin" ]; then
    printf 'cpu'
  elif command -v nvidia-smi >/dev/null 2>&1; then
    printf 'cuda'
  else
    printf 'cpu'
  fi
}

if ! docker info >/dev/null 2>&1; then
  echo "[FAIL] Docker is not running." >&2
  exit 1
fi

build=0
case "${1:-}" in
  --build|build) build=1 ;;
esac

app_port="$(env_or_default REGUMATE_APP_PORT 8000)"
frontend_port="$(env_or_default REGUMATE_DOCKER_FRONTEND_PORT 5174)"
qdrant_http_port="$(env_or_default REGUMATE_QDRANT_HTTP_PORT 6333)"
qdrant_grpc_port="$(env_or_default REGUMATE_QDRANT_GRPC_PORT 6334)"
torch_flavor="${REGUMATE_TORCH_FLAVOR:-$(default_torch_flavor)}"
app_image="${REGUMATE_APP_IMAGE:-regumate/app:dev-$torch_flavor}"
network_name="${REGUMATE_DOCKER_DEV_NETWORK:-regumate-dev}"

if [ "$torch_flavor" != "cuda" ] && [ "$torch_flavor" != "cpu" ]; then
  echo "[FAIL] REGUMATE_TORCH_FLAVOR must be 'cuda' or 'cpu'." >&2
  exit 1
fi

if [ "$build" -eq 1 ] || ! docker image inspect "$app_image" >/dev/null 2>&1; then
  echo "Building $app_image with REGUMATE_TORCH_FLAVOR=$torch_flavor..."
  docker build \
    --build-arg "REGUMATE_TORCH_FLAVOR=$torch_flavor" \
    --build-arg "REGUMATE_BUILD_ID=docker-dev" \
    -t "$app_image" \
    .
fi

mkdir -p data data/models data/qdrant
docker network inspect "$network_name" >/dev/null 2>&1 || docker network create "$network_name" >/dev/null

docker rm -f regumate-dev-frontend regumate-dev-app regumate-dev-qdrant >/dev/null 2>&1 || true

echo "Starting Qdrant dev container..."
docker run -d \
  --name regumate-dev-qdrant \
  --network "$network_name" \
  -p "127.0.0.1:$qdrant_http_port:6333" \
  -p "127.0.0.1:$qdrant_grpc_port:6334" \
  -v "$(pwd)/data/qdrant:/qdrant/storage" \
  qdrant/qdrant:v1.18.0 >/dev/null

deadline=$(( $(date +%s) + 60 ))
until curl -fsS "http://127.0.0.1:$qdrant_http_port/healthz" >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[FAIL] Qdrant did not become ready." >&2
    exit 1
  fi
  sleep 1
done

gpu_args=""
if [ "$torch_flavor" = "cuda" ] && docker run --rm --gpus all --entrypoint python "$app_image" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  echo "Docker GPU support detected; app dev container will use --gpus all."
  gpu_args="--gpus all"
else
  echo "Docker GPU support was not detected; app dev container will start in CPU fallback mode."
fi

env_file_args=""
if [ -f .env ]; then
  env_file_args="--env-file .env"
fi

cors_origins="http://localhost:$frontend_port,http://127.0.0.1:$frontend_port,http://localhost:$app_port,http://127.0.0.1:$app_port"

echo "Starting ReguMate backend dev container..."
# shellcheck disable=SC2086
docker run -d \
  --name regumate-dev-app \
  --network "$network_name" \
  $gpu_args \
  $env_file_args \
  -p "127.0.0.1:$app_port:8000" \
  -e "QDRANT_URL=http://regumate-dev-qdrant:6333" \
  -e "CORS_ORIGINS=$cors_origins" \
  -e "REGUMATE_FRONTEND_DEV_SERVER=http://regumate-dev-frontend:5174" \
  -e "MODEL_DEVICE=cuda" \
  -e "REGUMATE_BUILD_ID=docker-dev" \
  -v "$(pwd)/backend:/app/backend" \
  -v "$(pwd)/scripts:/app/scripts" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/data/models:/app/data/models:ro" \
  "$app_image" \
  python -m uvicorn backend.main:app --reload --reload-dir /app/backend --host 0.0.0.0 --port 8000 >/dev/null

echo "Starting ReguMate frontend dev container..."
docker run -d \
  --name regumate-dev-frontend \
  --network "$network_name" \
  -p "127.0.0.1:$frontend_port:5174" \
  -e "VITE_API_PROXY_TARGET=http://regumate-dev-app:8000" \
  -e "VITE_HMR_CLIENT_PORT=$frontend_port" \
  -v "$(pwd)/frontend:/app/frontend" \
  -v "regumate-frontend-node-modules:/app/frontend/node_modules" \
  -w /app/frontend \
  node:22-bookworm-slim \
  sh -c 'test -d node_modules/.vite || npm install && npm run dev -- --host 0.0.0.0 --port 5174' >/dev/null

deadline=$(( $(date +%s) + 90 ))
until curl -fsS "http://127.0.0.1:$app_port/api/health" >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:$frontend_port" >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[FAIL] ReguMate dev services did not become ready." >&2
    exit 1
  fi
  sleep 1
done

mkdir -p .run-state
cat > .run-state/docker-dev.json <<EOF
{"mode":"docker-run-dev","app_port":$app_port,"frontend_port":$frontend_port,"qdrant_http_port":$qdrant_http_port,"qdrant_grpc_port":$qdrant_grpc_port,"app_url":"http://127.0.0.1:$app_port","frontend_url":"http://127.0.0.1:$frontend_port","qdrant_url":"http://127.0.0.1:$qdrant_http_port","app_image":"$app_image","torch_flavor":"$torch_flavor"}
EOF
cp .run-state/docker-dev.json .run-state/runtime.json

echo
echo "ReguMate Docker development frontend: http://127.0.0.1:$frontend_port"
echo "ReguMate backend API: http://127.0.0.1:$app_port"
echo "Qdrant dashboard: http://127.0.0.1:$qdrant_http_port/dashboard"
echo
echo "Logs:"
echo "  docker logs -f regumate-dev-app"
echo "  docker logs -f regumate-dev-frontend"

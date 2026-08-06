#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/../.."

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

export REGUMATE_TORCH_FLAVOR="${REGUMATE_TORCH_FLAVOR:-$(default_torch_flavor)}"
export REGUMATE_APP_IMAGE="${REGUMATE_APP_IMAGE:-regumate/app:contest-$REGUMATE_TORCH_FLAVOR}"

if [ "$REGUMATE_TORCH_FLAVOR" != "cuda" ] && [ "$REGUMATE_TORCH_FLAVOR" != "cpu" ]; then
  echo "[FAIL] REGUMATE_TORCH_FLAVOR must be 'cuda' or 'cpu'." >&2
  exit 1
fi

compose_files="-f docker-compose.yml"
if [ "$REGUMATE_TORCH_FLAVOR" = "cuda" ] && command -v nvidia-smi >/dev/null 2>&1; then
  compose_files="$compose_files -f docker-compose.gpu.yml"
else
  echo "Starting without Docker GPU override; ReguMate will use CPU fallback if CUDA is unavailable."
fi

build_args=""
if [ "${1:-}" = "--build" ] || [ "${1:-}" = "build" ] || ! docker image inspect "$REGUMATE_APP_IMAGE" >/dev/null 2>&1; then
  build_args="--build"
fi

# shellcheck disable=SC2086
docker compose $compose_files up -d $build_args

app_port="${REGUMATE_APP_PORT:-8000}"
qdrant_port="${REGUMATE_QDRANT_HTTP_PORT:-6333}"
echo "ReguMate is starting at http://127.0.0.1:$app_port"
echo "Qdrant dashboard: http://127.0.0.1:$qdrant_port/dashboard"
echo "Logs: docker compose logs -f app"

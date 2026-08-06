#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/../.."

docker rm -f regumate-dev-frontend regumate-dev-app regumate-dev-qdrant >/dev/null 2>&1 || true
docker compose down
rm -f .run-state/docker-dev.json .run-state/runtime.json

echo "ReguMate Docker development containers and Compose services have been stopped."

#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
case "${1:-}" in
  ""|--gpu) ;;
  *) echo "Usage: sh scripts/offline/start.sh [--gpu]" >&2; exit 2 ;;
esac
if [ ! -f .env ]; then
  echo 'Required root .env is missing. Restore the complete submission package.' >&2
  exit 1
fi
test -f images/regumate-linux-amd64.tar || { echo 'Offline image archive is missing.' >&2; exit 1; }
docker load -i images/regumate-linux-amd64.tar
if [ "${1:-}" = --gpu ]; then
  docker compose -f docker-compose.offline.yml -f docker-compose.offline.gpu.yml up -d --no-build --pull never
  docker compose -f docker-compose.offline.yml -f docker-compose.offline.gpu.yml ps
else
  docker compose -f docker-compose.offline.yml up -d --no-build --pull never
  docker compose -f docker-compose.offline.yml ps
fi
echo 'Use the SERVICE_PORT configured in .env. Default address: http://127.0.0.1:8000'

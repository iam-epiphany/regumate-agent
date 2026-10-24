#!/bin/sh
set -eu
collection="${QDRANT_COLLECTION:-regumate_chunks}"
case "$collection" in
  ''|*[!A-Za-z0-9_-]*) echo 'Invalid QDRANT_COLLECTION.' >&2; exit 1 ;;
esac
if [ -f "/qdrant/storage/collections/$collection/config.json" ]; then
  exec ./qdrant --disable-telemetry
fi
exec ./qdrant --snapshot "/seed/regumate.snapshot:$collection" --disable-telemetry

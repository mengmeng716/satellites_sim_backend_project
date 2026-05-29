#!/usr/bin/env bash
set -euo pipefail

for c in sim-celery-worker sim-backend sim-redis sim-mysql; do
  if docker ps -a --format '{{.Names}}' | grep -q "^${c}$"; then
    docker rm -f "$c" >/dev/null
    echo "removed ${c}"
  fi
done

echo "done"

#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
  echo "[ERROR] .env not found in current directory: $PWD"
  echo "Please run: cp .env.example .env and edit as needed"
  exit 1
fi

set -a
source ./.env
set +a

DB_USER="${DB_USER:-root}"
DB_PASSWORD="${DB_PASSWORD:-123456}"
DB_NAME="${DB_NAME:-satellite_sim_db}"
DB_PORT_HOST="${DB_PORT_HOST:-3306}"
REDIS_PORT_HOST="${REDIS_PORT_HOST:-6379}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
DJANGO_DEBUG="${DJANGO_DEBUG:-False}"
CORS_ALLOWED_ORIGINS="${CORS_ALLOWED_ORIGINS:-}"

NET_NAME="sim-net"
MYSQL_CONTAINER="sim-mysql"
REDIS_CONTAINER="sim-redis"
BACKEND_CONTAINER="sim-backend"
CELERY_CONTAINER="sim-celery-worker"
FRONTEND_CONTAINER="sim-frontend"
BACKEND_IMAGE="satellites-sim-backend:1.0.0"
FRONTEND_IMAGE="satellite-frontend:1.0.0"

echo "[1/5] ensure docker network"
docker network inspect "$NET_NAME" >/dev/null 2>&1 || docker network create "$NET_NAME"

cleanup_container() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
    docker rm -f "$name" >/dev/null
  fi
}

echo "[2/5] remove old containers if exist"
cleanup_container "$CELERY_CONTAINER"
cleanup_container "$BACKEND_CONTAINER"
cleanup_container "$FRONTEND_CONTAINER"
cleanup_container "$REDIS_CONTAINER"
cleanup_container "$MYSQL_CONTAINER"

echo "[3/5] start mysql + redis"
docker run -d \
  --name "$MYSQL_CONTAINER" \
  --network "$NET_NAME" \
  --restart unless-stopped \
  -p "${DB_PORT_HOST}:3306" \
  -e MYSQL_ROOT_PASSWORD="${DB_PASSWORD}" \
  -e MYSQL_DATABASE="${DB_NAME}" \
  -v sim_mysql_data:/var/lib/mysql \
  mysql:8.0 \
  --default-authentication-plugin=mysql_native_password

docker run -d \
  --name "$REDIS_CONTAINER" \
  --network "$NET_NAME" \
  --restart unless-stopped \
  -p "${REDIS_PORT_HOST}:6379" \
  redis:7.2-alpine

echo "[4/5] start backend"
docker run -d \
  --name "$BACKEND_CONTAINER" \
  --network "$NET_NAME" \
  --restart unless-stopped \
  -p "${BACKEND_PORT}:8000" \
  -e DB_HOST="$MYSQL_CONTAINER" \
  -e DB_PORT=3306 \
  -e DB_USER="$DB_USER" \
  -e DB_PASSWORD="$DB_PASSWORD" \
  -e DB_NAME="$DB_NAME" \
  -e USE_REDIS=True \
  -e REDIS_HOST="$REDIS_CONTAINER" \
  -e REDIS_PORT=6379 \
  -e CELERY_BROKER_URL="redis://${REDIS_CONTAINER}:6379/0" \
  -e CELERY_RESULT_BACKEND="redis://${REDIS_CONTAINER}:6379/1" \
  -e DJANGO_DEBUG="$DJANGO_DEBUG" \
  -e CORS_ALLOWED_ORIGINS="$CORS_ALLOWED_ORIGINS" \
  "$BACKEND_IMAGE"

echo "[5/5] start celery worker"
docker run -d \
  --name "$CELERY_CONTAINER" \
  --network "$NET_NAME" \
  --restart unless-stopped \
  -e DB_HOST="$MYSQL_CONTAINER" \
  -e DB_PORT=3306 \
  -e DB_USER="$DB_USER" \
  -e DB_PASSWORD="$DB_PASSWORD" \
  -e DB_NAME="$DB_NAME" \
  -e USE_REDIS=True \
  -e REDIS_HOST="$REDIS_CONTAINER" \
  -e REDIS_PORT=6379 \
  -e CELERY_BROKER_URL="redis://${REDIS_CONTAINER}:6379/0" \
  -e CELERY_RESULT_BACKEND="redis://${REDIS_CONTAINER}:6379/1" \
  -e DJANGO_DEBUG="$DJANGO_DEBUG" \
  "$BACKEND_IMAGE" \
  celery -A satellites_sim_backend worker -l info

echo "[6/6] start frontend"
if docker image inspect "$FRONTEND_IMAGE" >/dev/null 2>&1; then
  docker run -d \
    --name "$FRONTEND_CONTAINER" \
    --network "$NET_NAME" \
    --restart unless-stopped \
    -p "${FRONTEND_PORT}:80" \
    -e API_HOST="$BACKEND_CONTAINER" \
    "$FRONTEND_IMAGE"
else
  echo "[WARN] frontend image not found: ${FRONTEND_IMAGE}, skip frontend container"
fi

echo "[OK] Deployment finished"
if docker ps --format '{{.Names}}' | grep -q "^${FRONTEND_CONTAINER}$"; then
  echo "Frontend URL: http://<target-ip>:${FRONTEND_PORT}"
fi
echo "Backend URL: http://<target-ip>:${BACKEND_PORT}"
echo "Check logs:"
if docker ps --format '{{.Names}}' | grep -q "^${FRONTEND_CONTAINER}$"; then
  echo "  docker logs -f ${FRONTEND_CONTAINER}"
fi
echo "  docker logs -f ${BACKEND_CONTAINER}"
echo "  docker logs -f ${CELERY_CONTAINER}"

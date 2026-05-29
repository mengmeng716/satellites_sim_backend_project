#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/offline_release_backend"
ARCHIVE_PATH="${PROJECT_DIR}/satellites_sim_backend_offline_deploy.tar.gz"

echo "[1/5] clean output dir"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "[2/5] build backend image (includes /app/data in image)"
cd "${PROJECT_DIR}"
if docker buildx version >/dev/null 2>&1; then
	echo "[build] using buildx + BuildKit"
	docker buildx build --load -t satellites-sim-backend:1.0.0 .
else
	echo "[build] buildx not found, fallback to docker build"
	if ! docker build -t satellites-sim-backend:1.0.0 .; then
		echo "[build] docker build failed, retrying with BuildKit disabled"
		DOCKER_BUILDKIT=0 docker build -t satellites-sim-backend:1.0.0 .
	fi
fi

echo "[3/5] pull infra images"
docker pull mysql:8.0
docker pull redis:7.2-alpine

echo "[4/5] export images"
docker save satellites-sim-backend:1.0.0 mysql:8.0 redis:7.2-alpine -o "${OUTPUT_DIR}/images.tar"

echo "[5/5] copy deployment files"
cp "${PROJECT_DIR}/docker-compose.offline.yml" "${OUTPUT_DIR}/docker-compose.yml"
cp "${PROJECT_DIR}/.env.offline.example" "${OUTPUT_DIR}/.env.example"
cp "${PROJECT_DIR}/deploy_no_compose.sh" "${OUTPUT_DIR}/deploy_no_compose.sh"
cp "${PROJECT_DIR}/stop_no_compose.sh" "${OUTPUT_DIR}/stop_no_compose.sh"
chmod +x "${OUTPUT_DIR}/deploy_no_compose.sh" "${OUTPUT_DIR}/stop_no_compose.sh"

cd "${PROJECT_DIR}"
tar -czf "${ARCHIVE_PATH}" -C "${PROJECT_DIR}" "$(basename "${OUTPUT_DIR}")"

echo "Done: ${ARCHIVE_PATH}"
echo "Deploy on target machine:"
echo "  tar -xzf satellites_sim_backend_offline_deploy.tar.gz"
echo "  cd offline_release_backend"
echo "  docker load -i images.tar"
echo "  cp .env.example .env"
echo "  docker compose up -d"

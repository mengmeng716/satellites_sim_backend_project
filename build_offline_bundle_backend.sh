#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
FRONTEND_DIR="${WORKSPACE_DIR}/satellite-front"
OUTPUT_DIR="${PROJECT_DIR}/offline_release_backend"
ARCHIVE_PATH="${PROJECT_DIR}/satellites_sim_backend_offline_deploy.tar.gz"
DOCKER_VERSION="${DOCKER_VERSION:-27.1.2}"
COMPOSE_VERSION="${COMPOSE_VERSION:-v2.29.7}"
DOCKER_TGZ_LOCAL="${DOCKER_TGZ_LOCAL:-}"
COMPOSE_BIN_LOCAL="${COMPOSE_BIN_LOCAL:-}"
INCLUDE_RUNTIME=1
MODE="full"
BUILD_FRONTEND=1
BUILD_BACKEND=1
INCLUDE_INFRA=1
TARGET_ARCH=""

need_cmd() {
	command -v "$1" >/dev/null 2>&1 || {
		echo "[ERROR] missing command: $1" >&2
		exit 1
	}
}

detect_arch() {
	local machine
	machine="$(uname -m)"
	normalize_arch "${machine}"
}

normalize_arch() {
	local machine="$1"
	case "${machine}" in
		x86_64|amd64)
			echo "x86_64"
			;;
		aarch64|arm64)
			echo "aarch64"
			;;
		*)
			echo "[ERROR] unsupported arch: ${machine}" >&2
			exit 1
			;;
	esac
}

copy_or_download_runtime() {
	local local_path="$1"
	local urls="$2"
	local dest="$3"
	local label="$4"
	local min_bytes="${5:-1}"

	if [[ -n "${local_path}" ]]; then
		[[ -f "${local_path}" ]] || {
			echo "[ERROR] ${label} local file not found: ${local_path}" >&2
			exit 1
		}
		cp "${local_path}" "${dest}"
		return
	fi

	local success=0
	local url
	IFS='|' read -r -a _url_list <<< "${urls}"
	for url in "${_url_list[@]}"; do
		if curl -fL --retry 2 --retry-delay 2 --connect-timeout 20 --max-time 600 "${url}" -o "${dest}"; then
			local file_size
			file_size="$(wc -c < "${dest}" 2>/dev/null || echo 0)"
			if [[ "${file_size}" -lt "${min_bytes}" ]]; then
				echo "[WARN] ${label} too small (${file_size} bytes), try next mirror: ${url}" >&2
				rm -f "${dest}"
				continue
			fi
			success=1
			break
		fi
		echo "[WARN] failed downloading ${label}, try next mirror: ${url}" >&2
	done

	if [[ "${success}" -ne 1 ]]; then
		echo "[ERROR] failed downloading ${label} from all mirrors" >&2
		exit 1
	fi
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--frontend-only)
			MODE="frontend-only"
			shift
			;;
		--backend-only)
			MODE="backend-only"
			shift
			;;
		--full)
			MODE="full"
			shift
			;;
		--without-runtime)
			INCLUDE_RUNTIME=0
			shift
			;;
		--target-arch)
			[[ $# -ge 2 ]] || { echo "[ERROR] --target-arch requires a value" >&2; exit 1; }
			TARGET_ARCH="$2"
			shift 2
			;;
		*)
			echo "[ERROR] unknown arg: $1" >&2
			echo "Usage: $0 [--full|--frontend-only|--backend-only] [--without-runtime] [--target-arch x86_64|aarch64]" >&2
			exit 1
			;;
	esac
done

case "${MODE}" in
	full)
		BUILD_FRONTEND=1
		BUILD_BACKEND=1
		INCLUDE_INFRA=1
		;;
	frontend-only)
		BUILD_FRONTEND=1
		BUILD_BACKEND=0
		INCLUDE_INFRA=0
		;;
	backend-only)
		BUILD_FRONTEND=0
		BUILD_BACKEND=1
		INCLUDE_INFRA=1
		;;
	*)
		echo "[ERROR] invalid mode: ${MODE}" >&2
		exit 1
		;;
esac

need_cmd docker
need_cmd tar
need_cmd sha256sum
if [[ "${INCLUDE_RUNTIME}" -eq 1 ]]; then
	need_cmd curl
fi

if [[ "${BUILD_FRONTEND}" -eq 1 ]]; then
	[[ -d "${FRONTEND_DIR}" ]] || { echo "[ERROR] frontend dir not found: ${FRONTEND_DIR}" >&2; exit 1; }
	[[ -f "${FRONTEND_DIR}/Dockerfile" ]] || { echo "[ERROR] frontend Dockerfile not found" >&2; exit 1; }
fi
if [[ "${BUILD_BACKEND}" -eq 1 ]]; then
	[[ -f "${PROJECT_DIR}/Dockerfile" ]] || { echo "[ERROR] backend Dockerfile not found" >&2; exit 1; }
	[[ -f "${PROJECT_DIR}/docker-compose.offline.yml" ]] || { echo "[ERROR] backend compose file not found" >&2; exit 1; }
fi

if [[ -n "${TARGET_ARCH}" ]]; then
	ARCH="$(normalize_arch "${TARGET_ARCH}")"
else
	ARCH="$(detect_arch)"
fi

echo "[info] runtime target arch: ${ARCH}"

echo "[1/7] clean output dir"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [[ "${BUILD_FRONTEND}" -eq 1 ]]; then
	echo "[2/7] build frontend dist + image"
	cd "${FRONTEND_DIR}"
	echo "[frontend] clean dist to avoid stale artifacts from other branches/builds"
	rm -rf dist
	if command -v pnpm >/dev/null 2>&1; then
		pnpm install --frozen-lockfile
		pnpm run build
	else
		npm install --legacy-peer-deps
		npm run build
	fi
	docker build -t satellite-frontend:1.0.0 .
else
	echo "[2/7] skip frontend build (mode=${MODE})"
fi

if [[ "${BUILD_BACKEND}" -eq 1 ]]; then
	echo "[3/7] build backend image (includes /app/data in image)"
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
else
	echo "[3/7] skip backend build (mode=${MODE})"
fi

if [[ "${INCLUDE_INFRA}" -eq 1 ]]; then
	echo "[4/7] pull infra images"
	docker pull mysql:8.0
	docker pull redis:7.2-alpine
else
	echo "[4/7] skip infra images (mode=${MODE})"
fi

echo "[5/7] export images"
IMAGES_TO_SAVE=()
if [[ "${BUILD_FRONTEND}" -eq 1 ]]; then
	IMAGES_TO_SAVE+=(satellite-frontend:1.0.0)
fi
if [[ "${BUILD_BACKEND}" -eq 1 ]]; then
	IMAGES_TO_SAVE+=(satellites-sim-backend:1.0.0)
fi
if [[ "${INCLUDE_INFRA}" -eq 1 ]]; then
	IMAGES_TO_SAVE+=(mysql:8.0 redis:7.2-alpine)
fi

if [[ "${#IMAGES_TO_SAVE[@]}" -eq 0 ]]; then
	echo "[ERROR] no images to export" >&2
	exit 1
fi

docker save "${IMAGES_TO_SAVE[@]}" -o "${OUTPUT_DIR}/images.tar"
(
	cd "${OUTPUT_DIR}"
	sha256sum "images.tar" > "images.tar.sha256"
)

echo "[6/7] copy deployment files"
if [[ "${BUILD_BACKEND}" -eq 1 ]]; then
	cp "${PROJECT_DIR}/docker-compose.offline.yml" "${OUTPUT_DIR}/docker-compose.yml"
	cp "${PROJECT_DIR}/.env.offline.example" "${OUTPUT_DIR}/.env.example"
	cp "${PROJECT_DIR}/deploy_no_compose.sh" "${OUTPUT_DIR}/deploy_no_compose.sh"
	cp "${PROJECT_DIR}/stop_no_compose.sh" "${OUTPUT_DIR}/stop_no_compose.sh"
	chmod +x "${OUTPUT_DIR}/deploy_no_compose.sh" "${OUTPUT_DIR}/stop_no_compose.sh"

	if [[ "${BUILD_FRONTEND}" -eq 0 ]]; then
		awk '
		BEGIN {in_frontend=0}
		{
			if ($0 ~ /^  frontend:[[:space:]]*$/) { in_frontend=1; next }
			if (in_frontend && $0 ~ /^  [a-zA-Z0-9_]+:[[:space:]]*$/) { in_frontend=0 }
			if (in_frontend) { next }
			print
		}
		' "${OUTPUT_DIR}/docker-compose.yml" > "${OUTPUT_DIR}/docker-compose.yml.tmp"
		mv "${OUTPUT_DIR}/docker-compose.yml.tmp" "${OUTPUT_DIR}/docker-compose.yml"
	fi
else
	cat > "${OUTPUT_DIR}/docker-compose.yml" <<'EOF'
version: '3.9'

services:
  frontend:
    image: satellite-frontend:1.0.0
    container_name: sim-frontend
    restart: unless-stopped
    ports:
      - "${FRONTEND_PORT:-5173}:80"
EOF
	cat > "${OUTPUT_DIR}/.env.example" <<'EOF'
FRONTEND_PORT=5173
EOF
	cat > "${OUTPUT_DIR}/deploy_frontend_no_compose.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

set -a
source ./.env
set +a

FRONTEND_PORT="${FRONTEND_PORT:-5173}"
docker rm -f sim-frontend >/dev/null 2>&1 || true
docker run -d --name sim-frontend --restart unless-stopped -p "${FRONTEND_PORT}:80" satellite-frontend:1.0.0
echo "[OK] Frontend started: http://<target-ip>:${FRONTEND_PORT}"
EOF
	chmod +x "${OUTPUT_DIR}/deploy_frontend_no_compose.sh"
fi

if [[ "${INCLUDE_RUNTIME}" -eq 1 ]]; then
	echo "[7/7] package offline Docker/Compose runtime"
	mkdir -p "${OUTPUT_DIR}/runtime/docker" "${OUTPUT_DIR}/runtime/compose"
	DOCKER_TGZ="${OUTPUT_DIR}/runtime/docker/docker-${DOCKER_VERSION}.tgz"
	COMPOSE_BIN="${OUTPUT_DIR}/runtime/compose/docker-compose-linux-${ARCH}"
	DOCKER_URLS="https://mirrors.aliyun.com/docker-ce/linux/static/stable/${ARCH}/docker-${DOCKER_VERSION}.tgz|https://download.docker.com/linux/static/stable/${ARCH}/docker-${DOCKER_VERSION}.tgz"
	COMPOSE_URLS="https://ghproxy.net/https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-${ARCH}|https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-${ARCH}"
	copy_or_download_runtime "${DOCKER_TGZ_LOCAL}" "${DOCKER_URLS}" "${DOCKER_TGZ}" "Docker runtime" 1000000
	copy_or_download_runtime "${COMPOSE_BIN_LOCAL}" "${COMPOSE_URLS}" "${COMPOSE_BIN}" "Compose plugin" 1000000
	chmod +x "${COMPOSE_BIN}"
	echo "${ARCH}" > "${OUTPUT_DIR}/runtime/TARGET_ARCH"
	(
		cd "${OUTPUT_DIR}/runtime/docker"
		sha256sum "docker-${DOCKER_VERSION}.tgz" > "docker-${DOCKER_VERSION}.tgz.sha256"
	)
	(
		cd "${OUTPUT_DIR}/runtime/compose"
		sha256sum "docker-compose-linux-${ARCH}" > "docker-compose-linux-${ARCH}.sha256"
	)

	cat > "${OUTPUT_DIR}/setup_runtime.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT_PREFIX=""
if [[ "${EUID}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
	 ROOT_PREFIX="sudo"
  else
	 echo "[ERROR] root privilege required (run as root or install sudo)" >&2
	 exit 1
  fi
fi

ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64|amd64) ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *) echo "[ERROR] unsupported arch: ${ARCH}" >&2; exit 1 ;;
esac

PACKAGE_ARCH="$(cat runtime/TARGET_ARCH 2>/dev/null || true)"
if [[ -z "${PACKAGE_ARCH}" ]]; then
	PACKAGE_ARCH="${ARCH}"
fi

if [[ "${PACKAGE_ARCH}" != "${ARCH}" ]]; then
	echo "[ERROR] runtime arch mismatch: package=${PACKAGE_ARCH}, host=${ARCH}" >&2
	echo "[HINT] rebuild bundle with --target-arch ${ARCH}" >&2
	exit 1
fi

DOCKER_TGZ="$(ls runtime/docker/docker-*.tgz 2>/dev/null | head -n1 || true)"
COMPOSE_BIN="runtime/compose/docker-compose-linux-${PACKAGE_ARCH}"

[[ -f "${DOCKER_TGZ}" ]] || { echo "[ERROR] docker tgz not found" >&2; exit 1; }
[[ -f "${COMPOSE_BIN}" ]] || { echo "[ERROR] compose binary not found" >&2; exit 1; }
command -v systemctl >/dev/null 2>&1 || {
	echo "[ERROR] systemctl not found; this installer currently supports systemd hosts only" >&2
	exit 1
}

mkdir -p runtime/tmp
tar -xzf "${DOCKER_TGZ}" -C runtime/tmp

${ROOT_PREFIX} install -m 0755 runtime/tmp/docker/* /usr/local/bin/
${ROOT_PREFIX} mkdir -p /usr/local/lib/docker/cli-plugins
${ROOT_PREFIX} install -m 0755 "${COMPOSE_BIN}" /usr/local/lib/docker/cli-plugins/docker-compose

if [[ ! -f /etc/systemd/system/docker.service ]]; then
cat > runtime/tmp/docker.service <<'UNIT'
[Unit]
Description=Docker Application Container Engine
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
ExecStart=/usr/local/bin/dockerd
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=2
TimeoutStartSec=0
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity
Delegate=yes
KillMode=process
OOMScoreAdjust=-500

[Install]
WantedBy=multi-user.target
UNIT
${ROOT_PREFIX} install -m 0644 runtime/tmp/docker.service /etc/systemd/system/docker.service
${ROOT_PREFIX} systemctl daemon-reload
fi

${ROOT_PREFIX} systemctl enable docker >/dev/null 2>&1 || true
${ROOT_PREFIX} systemctl restart docker

${ROOT_PREFIX} /usr/local/bin/docker --version >/dev/null
${ROOT_PREFIX} /usr/local/bin/docker compose version >/dev/null

ready=0
for _ in $(seq 1 20); do
	if ${ROOT_PREFIX} /usr/local/bin/docker info >/dev/null 2>&1; then
		ready=1
		break
	fi
	sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
	echo "[ERROR] docker daemon not ready after install" >&2
	echo "[HINT] check: systemctl status docker && journalctl -u docker -n 100" >&2
	exit 1
fi

echo "[DONE] docker/compose runtime installed"
EOF
	chmod +x "${OUTPUT_DIR}/setup_runtime.sh"
	bash -n "${OUTPUT_DIR}/setup_runtime.sh"
fi

cat > "${OUTPUT_DIR}/preflight_check.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

warn() { echo "[WARN] $*"; }
fail() { echo "[ERROR] $*"; exit 1; }
ok() { echo "[OK] $*"; }

ARCH="$(uname -m)"
case "${ARCH}" in
	x86_64|amd64) ARCH="x86_64" ;;
	aarch64|arm64) ARCH="aarch64" ;;
	*) fail "unsupported arch: ${ARCH}" ;;
esac
ok "arch=${ARCH}"

if [[ ! -f images.tar ]]; then
	fail "images.tar not found in current directory"
fi
if [[ -f images.tar.sha256 ]]; then
	sha256sum -c images.tar.sha256
	ok "images.tar sha256 verified"
else
	warn "images.tar.sha256 missing, skip checksum verification"
fi

if [[ -f runtime/TARGET_ARCH ]]; then
	PACKAGE_ARCH="$(cat runtime/TARGET_ARCH)"
	[[ "${PACKAGE_ARCH}" == "${ARCH}" ]] || fail "runtime arch mismatch: package=${PACKAGE_ARCH}, host=${ARCH}"
	ok "runtime arch matches (${PACKAGE_ARCH})"
else
	warn "runtime/TARGET_ARCH missing, skip runtime arch check"
fi

if command -v docker >/dev/null 2>&1; then
	ok "docker command found"
else
	warn "docker command not found, run ./setup_runtime.sh if this host has no docker"
fi

if command -v docker >/dev/null 2>&1; then
	if docker info >/dev/null 2>&1; then
		ok "docker daemon reachable"
	else
		warn "docker command exists but daemon not reachable"
	fi
fi

if command -v docker >/dev/null 2>&1; then
	if docker compose version >/dev/null 2>&1; then
		ok "docker compose plugin available"
	else
		warn "docker compose plugin unavailable; fallback to ./deploy_no_compose.sh"
	fi
fi

check_port() {
	local p="$1"
	if command -v ss >/dev/null 2>&1; then
		if ss -lnt "( sport = :${p} )" | grep -q LISTEN; then
			warn "port ${p} is already in use"
		else
			ok "port ${p} is free"
		fi
	else
		warn "ss not found, skip port check for ${p}"
	fi
}

check_port "${FRONTEND_PORT:-5173}"
check_port "${BACKEND_PORT:-8000}"
check_port "${DB_PORT_HOST:-3306}"
check_port "${REDIS_PORT_HOST:-6379}"

AVAIL_KB="$(df -Pk . | awk 'NR==2{print $4}')"
if [[ "${AVAIL_KB}" -lt 10485760 ]]; then
	warn "available disk is below 10GB"
else
	ok "disk space looks sufficient (>10GB)"
fi

echo "[DONE] preflight finished"
EOF
chmod +x "${OUTPUT_DIR}/preflight_check.sh"
bash -n "${OUTPUT_DIR}/preflight_check.sh"

cat > "${OUTPUT_DIR}/README_OFFLINE_DEPLOY.md" <<'EOF'
# Sim Backend + Frontend Offline Deploy

1. Extract package:
	tar -xzf satellites_sim_backend_offline_deploy.tar.gz
	cd offline_release_backend
2. If target host has no docker/compose:
	./setup_runtime.sh
3. Load images:
	./preflight_check.sh
	sha256sum -c images.tar.sha256
	docker load -i images.tar
4. Create env:
	cp .env.example .env
5. Start services:
	docker compose up -d
	# fallback: ./deploy_no_compose.sh (or ./deploy_frontend_no_compose.sh for frontend-only mode)

Notes:
- Target host must be Linux with systemd.
- Runtime package is architecture-specific (x86_64 or aarch64).
- If packager arch differs from target arch, build with: --target-arch <arch>

Frontend: http://<target-ip>:5173
Backend:  http://<target-ip>:8000
EOF

cd "${PROJECT_DIR}"
ARCHIVE_NAME="$(basename "${ARCHIVE_PATH}")"
tar -czf "${ARCHIVE_PATH}" -C "${PROJECT_DIR}" "$(basename "${OUTPUT_DIR}")"
(
	cd "${PROJECT_DIR}"
	sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256"
)

echo "Done: ${ARCHIVE_PATH}"
echo "SHA256: ${ARCHIVE_PATH}.sha256"
echo "Mode: ${MODE}"
echo "Runtime arch: ${ARCH}"
echo "Deploy on target machine:"
echo "  tar -xzf satellites_sim_backend_offline_deploy.tar.gz"
echo "  cd offline_release_backend"
echo "  (optional when docker/compose absent) ./setup_runtime.sh"
echo "  docker load -i images.tar"
echo "  cp .env.example .env"
echo "  docker compose up -d"

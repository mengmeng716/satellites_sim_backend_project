#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"

activate_venv() {
	if [[ -f "${HOME}/.venvs/django-sim/bin/activate" ]]; then
		# 与当前终端环境保持一致，优先 django-sim
		source "${HOME}/.venvs/django-sim/bin/activate"
	elif [[ -f "${HOME}/.venvs/django_sim/bin/activate" ]]; then
		source "${HOME}/.venvs/django_sim/bin/activate"
	else
		echo "[ERROR] 未找到虚拟环境，请先创建 ~/.venvs/django-sim 或 ~/.venvs/django_sim" >&2
		exit 1
	fi
}

wait_for_backend() {
	local url="http://127.0.0.1:8000/api/simulation/control/"
	local retries=30
	local i
	for i in $(seq 1 "${retries}"); do
		# 这个接口 POST 才 200，GET 返回 405 也说明服务已起来
		local code
		code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "${url}" || true)"
		if [[ "${code}" == "200" || "${code}" == "405" ]]; then
			echo "[OK] Django backend is ready (HTTP ${code})"
			return 0
		fi
		sleep 1
	done

	echo "[ERROR] Django backend startup timeout" >&2
	return 1
}

cleanup() {
	echo "正在清理后台测试进程..."
	if [[ -n "${CELERY_PID:-}" ]] && kill -0 "${CELERY_PID}" 2>/dev/null; then
		kill "${CELERY_PID}" || true
	fi
	if [[ -n "${DJANGO_PID:-}" ]] && kill -0 "${DJANGO_PID}" 2>/dev/null; then
		kill "${DJANGO_PID}" || true
	fi
	echo "清理完成，测试退出！"
}

trap cleanup EXIT INT TERM

activate_venv

# 补齐可能缺少的依赖
pip install websocket-client requests gevent >/dev/null

echo "▶ 启动 Celery Worker 中..."
celery -A satellites_sim_backend worker -l INFO -P gevent -c 100 > celery_test.log 2>&1 &
CELERY_PID=$!

echo "▶ 启动 Django ASGI Web 服务器 中..."
python manage.py runserver 0.0.0.0:8000 --noreload > django_test.log 2>&1 &
DJANGO_PID=$!

echo "================================================="
echo " 后台服务启动中 (Celery PID: ${CELERY_PID}, Django PID: ${DJANGO_PID})"
echo " 服务日志: celery_test.log / django_test.log"
echo "================================================="

if ! wait_for_backend; then
	echo "------ django_test.log (tail) ------"
	tail -n 80 django_test.log || true
	echo "------ celery_test.log (tail) ------"
	tail -n 80 celery_test.log || true
	exit 1
fi

echo "▶ 执行 websocket 自动化客户端测试 (test_client.py)..."
python test_client.py


# sudo service redis-server start
# service mysql status
# sudo service mysql start
# celery -A satellites_sim_backend worker -l info
# python manage.py runserver

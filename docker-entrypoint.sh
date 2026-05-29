#!/usr/bin/env bash
set -euo pipefail

DB_HOST="${DB_HOST:-mysql}"
DB_PORT="${DB_PORT:-3306}"
DB_USER="${DB_USER:-root}"
DB_PASSWORD="${DB_PASSWORD:-123456}"
DB_NAME="${DB_NAME:-satellite_sim_db}"

# 等待 MySQL 就绪
for i in $(seq 1 60); do
  if mysqladmin ping -h"${DB_HOST}" -P"${DB_PORT}" -u"${DB_USER}" -p"${DB_PASSWORD}" --silent >/dev/null 2>&1; then
    break
  fi
  echo "[entrypoint] waiting for MySQL ${DB_HOST}:${DB_PORT} (${i}/60) ..."
  sleep 2
done

# 自动建库（仅不存在时创建；存在则不影响）
mysql -h"${DB_HOST}" -P"${DB_PORT}" -u"${DB_USER}" -p"${DB_PASSWORD}" -e \
  "CREATE DATABASE IF NOT EXISTS \`${DB_NAME}\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 迁移结构，不导入 init_data
python manage.py migrate --noinput

exec "$@"

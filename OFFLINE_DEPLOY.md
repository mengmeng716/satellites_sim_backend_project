# satellites_sim_backend_project 离线部署说明

## 目标
- 后端离线部署（Django + Celery + MySQL + Redis）
- `data/` 目录随镜像打包（用户不需要单独看到或拷贝）
- 不导入 `init_data.sql`
- 目标机器上：数据库不存在则自动创建，存在则直接复用

## 一、在有网机器打包
在项目根目录执行：

```bash
cd /home/qmm/workspace/satellites_sim_backend_project
./build_offline_bundle_backend.sh
```

产物：
- `satellites_sim_backend_offline_deploy.tar.gz`

该包包含：
- `images.tar`（backend/mysql/redis 镜像）
- `docker-compose.yml`
- `.env.example`
- `deploy_no_compose.sh`
- `stop_no_compose.sh`

说明：`data/` 已通过 Dockerfile 中 `COPY . /app/` 进入 backend 镜像。

## 二、在离线机器部署

先完成解包与镜像导入（两种方式共用）：

```bash
tar -xzf satellites_sim_backend_offline_deploy.tar.gz
cd offline_release_backend
docker load -i images.tar
cp .env.example .env
# 按需修改 .env（端口、密码等）
```

### 方式 A：使用 Docker Compose

```bash
docker compose up -d
```

### 方式 B：不使用 Docker Compose（仅 docker run）

```bash
chmod +x deploy_no_compose.sh stop_no_compose.sh
./deploy_no_compose.sh
```

停止（无 compose 方式）：

```bash
./stop_no_compose.sh
```

## 三、数据库策略（默认）
backend/celery 启动时会执行 `docker-entrypoint.sh`：
1. 等待 MySQL 可连接
2. 执行 `CREATE DATABASE IF NOT EXISTS <DB_NAME>`
3. 执行 `python manage.py migrate --noinput`
4. 启动目标进程（Django 或 Celery）

因此：
- 数据库不存在：自动创建并迁移结构
- 数据库存在：直接使用，不会导入 `init_data.sql`

## 四、验证

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f celery_worker
```

若使用无 compose 方式，改为：

```bash
docker ps
docker logs -f sim-backend
docker logs -f sim-celery-worker
```

访问后端：
- `http://<目标机IP>:8000`

## 五、升级/重打包
代码变更后重新执行：

```bash
./build_offline_bundle_backend.sh
```

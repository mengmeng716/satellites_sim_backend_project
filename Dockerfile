# 1. 指定 Python 3.12 基础镜像（尽量与你开发环境一致）
FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

# 安装 MySQL 客户端构建依赖（mysqlclient 需要 pkg-config 和 C 编译器相关组件）
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    default-libmysqlclient-dev \
    default-mysql-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 先复制依赖列表并安装（利用国内清华源加速，避免网络报错）
COPY requirements.txt /app/
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 4. 将后端项目的所有代码拷贝到容器中
COPY . /app/

# 启动入口：等待数据库、自动建库(不存在时)、执行 migrate，再启动主进程
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# 专门为 PyTorch 配置源，以获取带 CUDA 11.8 后缀的版本
RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 5. 暴露 Django 默认运行的 8000 端口
EXPOSE 8000

# 6. 启动后端服务
# 说明：默认启动 Django；在 docker compose 里可通过 command 覆盖为 Celery。
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
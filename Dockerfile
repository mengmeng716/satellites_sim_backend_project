# 1. 指定 Python 3.12 基础镜像（尽量与你开发环境一致）
FROM python:3.12-slim

# 安装 MySQL 客户端构建依赖（mysqlclient 需要 pkg-config 和 C 编译器相关组件）
RUN apt-get update && apt-get install -y \
    pkg-config \
    default-libmysqlclient-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 先复制依赖列表并安装（利用国内清华源加速，避免网络报错）
COPY requirements.txt /app/
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 4. 将后端项目的所有代码拷贝到容器中
COPY . /app/

# 专门为 PyTorch 配置源，以获取带 CUDA 11.8 后缀的版本
RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 5. 暴露 Django 默认运行的 8000 端口
EXPOSE 8000

# 6. 启动后端服务
# 注意：这里用的是开发服务器，生产环境规范做法推荐改成 gunicorn，例如 gunicorn satellites_sim_backend.wsgi -b 0.0.0.0:8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
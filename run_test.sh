#!/bin/bash

# 1. 激活虚拟环境
source ~/.venvs/django_sim/bin/activate

# 2. 补齐可能缺少的依赖
pip install websocket-client requests gevent

# 3. 后台起 Celery (输出丢进 celery.log 以防刷乱终端。如果有报错可以看这个日志)
echo "▶ 启动 Celery Worker 中..."
celery -A satellites_sim_backend worker -l INFO -P gevent -c 100 > celery_test.log 2>&1 &
CELERY_PID=$!

# 4. 后台起 Django Asgi 
echo "▶ 启动 Django ASGI Web 服务器 中..."
python manage.py runserver 0.0.0.0:8000 > django_test.log 2>&1 &
DJANGO_PID=$!

echo "================================================="
echo " 后台服务已就绪！ (Celery PID: $CELERY_PID, Django PID: $DJANGO_PID)"
echo " 服务运行日志写在: celery_test.log 与 django_test.log 中"
echo "================================================="
echo "稍等2秒让服务初始化..."
sleep 5

# 5. 执行测试客户端
echo "▶ 执行 websocket 自动化客户端测试 (test_client.py)..."
python test_client.py

# ================= 退出清理 =====================
# 当 test_client.py (Ctrl+C) 结束时，把刚才挂后台的服务杀掉，保证不占端口
echo "正在清理后台测试进程..."
kill $CELERY_PID
kill $DJANGO_PID
echo "清理完成，测试退出！"


# redis-server
# service mysql status
# sudo service mysql start
# celery -A satellites_sim_backend worker -l info
# python manage.py runserver

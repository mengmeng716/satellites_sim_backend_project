import os
from celery import Celery

# 设定 Django的默认环境变量
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'satellites_sim_backend.settings')

app = Celery('satellites_sim_backend')

# 使用字符串让 worker 自动加载 django settings 中的配置
# namespace='CELERY' 表示所有的 celery 相关的配置键在 setting 中都必须有前缀 'CELERY_'
app.config_from_object('django.conf:settings', namespace='CELERY')

# 从所有的注册到 Django 的应用中自动搜寻 task 任务
app.autodiscover_tasks()

# 显式导入那些不在 tasks.py 里的非标准定义的任务模块
app.autodiscover_tasks(['simulation_api'], related_name='db_services')

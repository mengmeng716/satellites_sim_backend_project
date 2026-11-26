"""转发 :mod:`src.config`。

本目录下的模块以 ``import config`` 扁平方式导入配置；为保证全工程只有一份配置
（激光器预设、链路额定速率、星座参数等），这里直接转发 ``src.config``。
"""

from src.config import *  # noqa: F401,F403

# -*- coding: utf-8
import os.path


class FileLog:
    def __init__(self, log_dir=''):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log_dir = log_dir
        # gen据当前时间创建文件

    # 记录数据
    def log(self, data):

        pass

    # 获取记录器

    @property
    def logger(self):

        return

    # 从日志文件中获取日志数据
    def get_log(self):
        pass
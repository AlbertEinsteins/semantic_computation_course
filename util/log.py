# -*- coding: utf-8
import os.path
import logging
import time


class FileLog:
    def __init__(self, filename=None):
        # 根据当前时间创建文件
        assert filename is not None, 'filename is required.'
        self.filename = filename

        self.logger = self.get_logger()

    # 记录数据
    def log(self, info_str):
        self.logger.info(info_str)

    # 获取记录器
    def get_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                      datefmt="%a %b %d %H:%M:%S %Y")
        # 打印到控制台
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)

        # file handler
        log_dir = os.path.join(os.path.join(os.getcwd(), 'running'),
                               time.strftime('%Y-%m-%d-%H', time.localtime()))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(os.path.join(log_dir, self.filename), mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    # 从日志文件中获取日志数据
    def get_log(self):
        pass

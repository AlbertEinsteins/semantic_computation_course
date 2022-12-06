import time
import threading

class MultiThread:
    """
    多线层提取一个列表，为列表每个元素分配一个线程
    @list: 待操作的列表
    @work: 工作函数
    :rtn: 列表，长度和list一样，外部传入，存储线程的处理结果
    """
    def __init__(self, list, work, rtn):
        self.list = list
        self.work = work
        self.num_workers = len(list)

        self.rtn = rtn

    def do_work(self):
        """启动多个线程处理list"""
        thread_list = []
        for i in range(self.num_workers):
            thread_list.append(threading.Thread(name='worker-{}'.format(i),
                                                target=self.work, args=(i, self.list[i], self.rtn)))
        # 启动
        for i in range(self.num_workers):
            thread_list[i].start()
            time.sleep(0.1)

        for i in range(self.num_workers):
            thread_list[i].join()

    def get_return(self):
        return self.rtn

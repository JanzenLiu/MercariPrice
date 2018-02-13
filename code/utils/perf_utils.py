# wrapper: log time and memory consumption
import time
import os
import psutil
import numpy as np


def get_duration_str(secs):
    if secs <= 60:
        ret = "{} seconds".format(round(secs, 1))
    elif secs <= 60*60:
        ret = "{} minutes".format(round(secs/60.0, 1))
    else:
        ret = "{} hours".format(round(secs/3600.0, 1))
    return ret


def get_size_str(num_bytes, units=None):
    assert num_bytes >= 0, "Negative size is not allowed"
    if not units:
        units = ['B', 'KB', 'MB', 'GB']
    return "{:.2f}{}".format(num_bytes, units[0]) if num_bytes < 1024 else get_size_str(num_bytes/1024.0, units[1:])


def get_memory_bytes():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def get_memory_str():
    return get_size_str(get_memory_bytes())


def get_size_diff_str(num_bytes):
    sign = '+' if num_bytes >= 0 else '-'
    return "{}{}".format(sign, get_size_str(abs(num_bytes)))


def get_eval_str(scores):
    return "{:.6}(+/-{:.6})".format(np.mean(scores), np.std(scores))


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def task(task_name):
    def monitor_perf(func):
        def func_wrapper(*args, **kw):
            print("[{}] running task \"{}\"...".format(
                time.strftime("%z %Y-%m-%d %H:%M:%S", time.localtime()), task_name))
            t_start = time.time()
            m_start = get_memory_bytes()
            ret = func(*args, **kw)
            m_end = get_memory_bytes()
            msg_end = "[{}] finish task \"{}\".".format(
                time.strftime("%z %Y-%m-%d %H:%M:%S", time.localtime()), task_name)
            msg_time = 'time consumption: {}.'.format(get_duration_str(time.time() - t_start))
            msg_mem = 'memory usage: {}({}).'.format(get_size_str(m_end), get_size_diff_str(m_end - m_start))
            print(msg_end, msg_time, msg_mem)
            return ret
        return func_wrapper
    return monitor_perf

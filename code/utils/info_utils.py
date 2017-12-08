import psutil
import os
import numpy as np


def get_duration_str(secs):
    if secs <= 60:
        ret = "{} seconds".format(round(secs, 1))
    elif secs <= 3600:
        ret = "{} minutes".format(round(secs/60.0, 1))
    else:
        ret = "{} hours".format(round(secs/3600.0, 1))
    return ret


def get_size_str(num_bytes, units=None):
    assert(num_bytes>=0, "Negative size is not allowd")
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

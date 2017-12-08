import time
from . import info_utils


# run a function with printing its execution time and memory usage
def run_func(func, **params):
    # TODO: to rewrite as a decorator
    # TODO: to ignore process idle time
    t_ = time.time()
    m_ = info_utils.get_memory_bytes()
    ret = func(**params)
    m2_ = info_utils.get_memory_bytes()
    print('time consumption: {}'.format(info_utils.get_duration_str(time.time() - t_)))
    print('memory usage: {}({})'.format(info_utils.get_size_str(m2_), info_utils.get_size_diff_str(m2_ - m_)))
    print()
    return ret

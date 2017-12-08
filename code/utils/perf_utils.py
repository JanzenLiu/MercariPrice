import time


def run_func(func, **params):
    # todo: do not take idle time into account
    _t = time.time()  # start time
    ret = func(**params)
    return ret, time.time()-_t  # original return and duration

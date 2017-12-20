import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


# Copied from: https://www.kaggle.com/marknagelberg/rmsle-function
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


def get_stats(nums):
    return [np.mean(nums), np.std(nums), np.min(nums), np.percentile(nums, 25),
            np.percentile(nums, 50), np.percentile(nums, 75), np.max(nums)]


def gen_normal_data_with_noise(shape=10000, x_mean=0, x_std=1, noise_mean=0, noise_std=0.1):
    x = np.random.randn(shape) * x_std + x_mean
    noise = np.random.randn(shape) * noise_std + noise_mean
    x_disturbed = x + noise
    return x, x_disturbed, noise


def plot_diff(x, y, callback=None, *args, **params):
    # TODO: to restrict input to one-dimensional arrays, i.e. with shape of (n) or (n, )
    assert x.shape == y.shape, "Inconsistent data shape: {} and {}".format(x.shape, y.shape)
    if not args:
        args = ['bo']
    params['alpha'] = params.pop('alpha', 0.2)
    plt.plot(x, y-x, *args, **params)
    if callback:
        callback()
    plt.show()

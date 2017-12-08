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

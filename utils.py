import numpy as np


def softmax(arr):
    e = np.e ** arr
    return e / e.sum()
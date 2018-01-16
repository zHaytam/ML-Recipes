import math


def mean(array):
    """
    Returns the mean (average) of a numpy array
        :param array: A one dimensional numpy array
    """

    return float(sum(array) / max(len(array), 1))


def sd(array, sample=True):
    """
    Returns the (sample) standard deviation of a numpy array
        :param array: A one dimensional numpy array
        :param sample: A boolean indicating if we want the sample standard deviation
    """

    if not array.any():
        raise ValueError('The array cannot be empty.')

    arr_mean = mean(array)
    div = len(array) - 1 if sample else len(array)
    return math.sqrt(sum((array - arr_mean) ** 2) / div)

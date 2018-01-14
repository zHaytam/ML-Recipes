import math


def euclidean_distance(point1, point2):
    """
    returns the euclidean distance between 2 points
        :param point1: The first point (an array of integers)
        :param point2: The second point (an array of integers)
    """
    return math.sqrt(sum((point1 - point2) ** 2))

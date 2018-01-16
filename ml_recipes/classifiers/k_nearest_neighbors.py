from collections import defaultdict
from ..math_utils import euclidean_distance


class KNearestNeighborsClassifier:

    def __init__(self):
        self.x_train = []
        self.y_train = []

    def fit(self, x_train, y_train):
        """
        Sets the training data
            :param self: ~
            :param x_train: The features of the training data
            :param y_train: The labels of the training data
        """

        if len(x_train) != len(y_train):
            raise ValueError(
                'The length of the training data (X and Y) must be the same.')

        if not x_train.any():
            raise ValueError('Cannot work with an empty training data.')

        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, k=3):
        """
        Returns an array of the predicted labels for the test data
            :param self: ~
            :param x_test: The test features
            :param k: The number of neighbors to take in consideration
        """

        if not self.x_train.any() or not self.y_train.any():
            raise ValueError('Invalid training data.')

        predictions = []

        for x_row in x_test:
            distances = self.__get_sorted_distances(x_row)[:k]
            labels = defaultdict(int)

            for dist in distances:
                labels[dist[1]] += 1

            predictions.append(sorted(labels)[-1])

        return predictions

    def __get_sorted_distances(self, x_row):
        """
        Returns a sorted array of the distances and the labels they represent
            :param self: ~
            :param x_row: The point (features)
        """

        distances = []

        for n in range(0, len(self.x_train)):
            tup = (euclidean_distance(self.x_train[n], x_row), self.y_train[n])
            distances.append(tup)

        return sorted(distances)

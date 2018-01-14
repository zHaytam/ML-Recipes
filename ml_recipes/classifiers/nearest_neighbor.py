from .classifier import Classifier
from ..math_utils import euclidean_distance


class NearestNeighborClassifier(Classifier):

    def __init__(self):
        self.x_train = []
        self.y_train = []

    def fit(self, x_train, y_train):
        if len(x_train) != len(y_train):
            raise ValueError(
                'The length of the training data (X and Y) must be the same.')

        if not x_train.any():
            raise ValueError('Cannot work with an empty training data.')

        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        if not self.x_train.any() or not self.y_train.any():
            raise ValueError('Invalid training data.')

        predictions = []

        for row in x_test:
            predictions.append(self.find_closest_y(row))

        return predictions

    def find_closest_y(self, row):
        closest_dist = euclidean_distance(row, self.x_train[0])
        closest_index = 0
        for n in range(1, len(self.x_train)):
            temp_dist = euclidean_distance(row, self.x_train[n])
            if temp_dist < closest_dist:
                closest_dist = temp_dist
                closest_index = n

        return self.y_train[closest_index]

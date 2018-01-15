from ..math_utils import euclidean_distance


class NearestNeighbourClassifier:

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

    def predict(self, x_test):
        """
        Returns an array of the predicted labels for the test data
            :param self: 
            :param x_test: The test features
        """
        if not self.x_train.any() or not self.y_train.any():
            raise ValueError('Invalid training data.')

        predictions = []

        for x_row in x_test:
            predictions.append(self.__find_closest_y(x_row))

        return predictions

    def __find_closest_y(self, x_row):
        """
        Returns the closest label to a certain point
            :param self: 
            :param x_row: A point (features)
        """
        closest_dist = euclidean_distance(x_row, self.x_train[0])
        closest_index = 0
        for n in range(1, len(self.x_train)):
            temp_dist = euclidean_distance(x_row, self.x_train[n])
            if temp_dist < closest_dist:
                closest_dist = temp_dist
                closest_index = n

        return self.y_train[closest_index]

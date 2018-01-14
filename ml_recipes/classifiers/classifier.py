from abc import ABC, abstractmethod


class Classifier(ABC):

    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml_recipes.classifiers.nearest_neighbour import NearestNeighbourClassifier
from ml_recipes.classifiers.k_nearest_neighbors import KNearestNeighborsClassifier
from ml_recipes.linear_model.simple_linear_regression import SimpleLinearRegression
# from ml_recipes.ml_utils import std

IRIS_DATASET = datasets.load_iris()
FEATURES = IRIS_DATASET.data
LABELS = IRIS_DATASET.target


def test_on_iris_nn(x_train, x_test, y_train, y_test):
    nnc = NearestNeighbourClassifier()
    nnc.fit(x_train, y_train)
    y_pred = nnc.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


def test_on_iris_knn(x_train, x_test, y_train, y_test):
    knnc = KNearestNeighborsClassifier()
    knnc.fit(x_train, y_train)
    y_pred = knnc.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


def test_on_iris():
    x_train, x_test, y_train, y_test = train_test_split(
        FEATURES, LABELS, test_size=0.5)

    test_on_iris_nn(x_train, x_test, y_train, y_test)
    test_on_iris_knn(x_train, x_test, y_train, y_test)


def test_simple_linear_regression():
    points = np.genfromtxt('data/cricket_chirps_vs_temperature.txt',
                           dtype=float, delimiter=';')

    slr = SimpleLinearRegression()
    slr.fit(points)
    print(slr.predict(20))


# tests
# test_on_iris()
# print(std(np.array([1, 2, 3, 4, 5])))
test_simple_linear_regression()

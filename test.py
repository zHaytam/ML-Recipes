from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml_recipes.classifiers.nearest_neighbor import NearestNeighborClassifier

IRIS_DATASET = datasets.load_iris()
FEATURES = IRIS_DATASET.data
LABELS = IRIS_DATASET.target


def test_on_iris():
    x_train, x_test, y_train, y_test = train_test_split(
        FEATURES, LABELS, test_size=0.5)

    nnc = NearestNeighborClassifier()
    nnc.fit(x_train, y_train)
    y_pred = nnc.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


test_on_iris()

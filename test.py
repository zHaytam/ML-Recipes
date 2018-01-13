from sklearn import datasets

IRIS_DATASET = datasets.load_iris()
FEATURES = IRIS_DATASET.data
LABELS = IRIS_DATASET.target

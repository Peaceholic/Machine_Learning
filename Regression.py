from mglearn import datasets
from sklearn.datasets import load_boston

boston = load_boston()
print("데이터의 형태: {}".format(boston.data.shape))

X, y = datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))

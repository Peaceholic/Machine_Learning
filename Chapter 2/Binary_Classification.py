import mglearn
from mglearn import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np

X, y = datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["클래스 0", "클래스 1"], loc=4)
plt.xlabel("첫 번째 특성")
plt.ylabel("두 번째 특성")

cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("유방암 데이터의 형태: {}".format(cancer.data.shape))
print("클래스별 샘플 개수:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("특성 이름:\n{}".format(cancer.feature_names))

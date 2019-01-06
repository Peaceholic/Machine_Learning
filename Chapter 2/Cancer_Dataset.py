from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("format of cancer data: {}".format(cancer.data.shape))
print("num of samples in each class:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
))
print("name of attributes: \n{}".format(cancer.feature_names))

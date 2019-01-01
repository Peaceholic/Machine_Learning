import numpy as np

X = np.array([[0, 1, 0, 1],
             [1, 0, 1, 1],
             [0, 0, 0, 1],
             [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
    # repeated in each class
    # count the num of 1 in each attribute
    counts[label] = X[y == label].sum(axis=0)
print("Feature count:\n{}".format(counts))

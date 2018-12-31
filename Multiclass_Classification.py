from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

X, y = make_blobs(random_state=42)

#mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("Attribute 0")
# plt.ylabel("Attribute 1")
# plt.legend(["Class 0", "Class 1", "Class 2"])

linear_svm = LinearSVC().fit(X, y)

print("Coefficient list size: ", linear_svm.coef_.shape)
print("Intercept list size: ", linear_svm.intercept_.shape)

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15,15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

# plt.ylim(-10,15)
# plt.xlim(-10,8)
plt.xlabel("Attribute 0")
plt.ylabel("Attribute 1")
plt.legend(["Class 0", "Class 1", "Class 2", "Class 0 Boundary", "Class 1 Boundary",
            "Class 2 Boundary"], loc=(1.01, 0.3))

plt.show()

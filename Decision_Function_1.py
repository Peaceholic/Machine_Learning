import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

y_named = np.array(["blue", "red"])[y]

X_train, X_test, y_train_named, y_test_named, y_train, y_test = \
    train_test_split(X, y_named, y, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

print("X_test.shape: {}".format(X_test.shape))
print("Decision Function Result Format: {}".format(gbrt.decision_function(X_test).shape))
print("Decision Function:\n{}".format(gbrt.decision_function(X_test)[:6]))

print("Comparision:\n{}".format(gbrt.decision_function(X_test) > 0))
print("Estimation:\n{}".format(gbrt.predict(X_test)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,
                                fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
                                            alpha=.4, cm=mglearn.ReBl)

for ax in axes:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
                             markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
                             markers='o', ax=ax)
    ax.set_xlabel("Attribute 0")
    ax.set_ylabel("Attribute 1")

cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["Test Class 0", "Test Class 1", "Train Class 0",
                "Train Class 1"], ncol=4, loc=(.1, 1.1))

plt.show()

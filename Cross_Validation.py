from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris= load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)

print("Cross Validation Score: {}".format(scores))
print("Mean Score: {:.2f}".format(scores.mean()))

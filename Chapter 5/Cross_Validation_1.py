from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit

iris= load_iris()
logreg = LogisticRegression()
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
loo = LeaveOneOut()
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)

# scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
# scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)

print("Cross Validation Score: {}".format(scores))
print("Mean Score: {:.2f}".format(scores.mean()))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=42
)

gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)

print("Decision function result shape: {}".format(gbrt.decision_function(X_test).shape))
print("Decision function result:\n{}".format(gbrt.decision_function(X_test)[:6, :]))

print("The largest decision function index: \n{}".format(
    np.argmax(gbrt.decision_function(X_test), axis=1)
))
print("Prediction:\n{}".format(gbrt.predict(X_test)))

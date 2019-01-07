from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=4
)

param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

pipe = make_pipeline(StandardScaler(), LogisticRegression())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best model:\n{}".format(grid.best_estimator_))
print("Logistic Regression Step:\n{}".format(grid.best_estimator_.named_steps["logisticregression"]))
print("Logistic Regression Coefficient:\n{}".format(grid.best_estimator_.named_steps["logisticregression"].coef_))


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)

param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

pipeline = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipeline.fit(X_train, y_train)

grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print("Optimal Cross Validation Score: {:.2f}".format(grid.best_score_))
print("Test Set Score: {:.2f}".format(grid.score(X_test, y_test)))
print("Optimal parameter: {}".format(grid.best_params_))

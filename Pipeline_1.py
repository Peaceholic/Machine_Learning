from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)

# scaler = MinMaxScaler().fit(X_train)
#
# X_train_scaled = scaler.transform(X_train)
#
# svm = SVC()
# svm.fit(X_train_scaled, y_train)

pipeline = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
pipeline.fit(X_train, y_train)

# X_test_scaled = scaler.transform(X_test)
# # print("Test Score: {:.2f}".format(svm.score(X_test_scaled, y_test)))

print("Test Score: {:.2f}".format(pipeline.score(X_test, y_test)))

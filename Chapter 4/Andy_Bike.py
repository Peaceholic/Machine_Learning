from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

citibike = mglearn.datasets.load_citibike()

y = citibike.values
X = citibike.index.astype("int64").values.reshape(-1,1)

n_train = 184

def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test =  target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("Test Set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))

    plt.xticks(range(0, len(X), 8), rotation=90, ha="left")

    plt.plot(range(n_train), y_train, label="Train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="Test")
    plt.plot(range(n_train), y_pred_train, '--', label="Train Prediction")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="Test Prediction")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Num of rent")

    plt.show()

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
enc = OneHotEncoder()
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

X_hour = citibike.index.hour.values.reshape(-1,1)
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1,1),
                         citibike.index.hour.values.reshape(-1,1)])
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)

lr = Ridge()

eval_on_features(X_hour_week_onehot_poly, y, lr)

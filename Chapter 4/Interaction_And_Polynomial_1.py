import numpy as np
import mglearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1,1)

bins = np.linspace(-3, 3, 11)

which_bind = np.digitize(X, bins=bins)

encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bind)

poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)

X_binned = encoder.transform(which_bind)
X_combined = np.hstack([X, X_binned])
X_product = np.hstack([X_binned, X*X_binned])
X_poly = poly.transform(X)

line_binned = encoder.transform(np.digitize(line, bins=bins))

# reg = LinearRegression().fit(X_combined, y)
#
# line_combined = np.hstack([line, line_binned])
# plt.plot(line, reg.predict(line_combined), label='Linear Regression with original feature')
#
# for bin in bins:
#     plt.plot([bin, bin], [-3,3], ':', c='k', linewidth=1)
# plt.legend(loc="best")
# plt.ylabel("Regression Output")
# plt.xlabel("Input Feature")
# plt.plot(X[:, 0], y, 'o', c='k')

# reg = LinearRegression().fit(X_product, y)
#
# line_product = np.hstack([line_binned, line*line_binned])
#
# plt.plot(line, reg.predict(line_product), label='Linear Regression with original feature multiplication')
#
# for bin in bins:
#     plt.plot([bin, bin], [-3,3], ':', c='k', linewidth=1)
# plt.legend(loc="best")
# plt.ylabel("Regression Output")
# plt.xlabel("Input Feature")
# plt.plot(X[:, 0], y, 'o', c='k')

reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label="Polynomial Regression")
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression Output")
plt.xlabel("Input Feature")
plt.legend(loc="best")

plt.show()

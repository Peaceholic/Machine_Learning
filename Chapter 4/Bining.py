import numpy as np
import mglearn
from sklearn.preprocessing import OneHotEncoder

X, y = mglearn.datasets.make_wave(n_samples=100)

bins = np.linspace(-3, 3, 11)
print("Section: {}".format(bins))

which_bind = np.digitize(X, bins=bins)
print("\nData Point:\n", X[:5])
print("\nBelonging Section:\n", which_bind[:5])

encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bind)

X_binned = encoder.transform(which_bind)
print(X_binned[:5])

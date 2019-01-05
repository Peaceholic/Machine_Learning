from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()
scaler = StandardScaler()
pca = PCA(n_components=2)

scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

print("Original Data Format: {}".format(str(X_scaled.shape)))
print("Reduced Data Format: {}".format(str(X_pca.shape)))

plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(["Negative", "Positive"], loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("1st main feature")
plt.ylabel("2nd main feature")

plt.show()

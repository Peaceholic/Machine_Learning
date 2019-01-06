import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, ward

X, y =make_blobs(random_state=0, n_samples=12)

linkage_array = ward(X)
dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, '2 of clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, '3 of clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample #")
plt.ylabel("Clustering distance")

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

S = mglearn.datasets.make_signals()

A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)

print("Data Format: {}".format(X.shape))

nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)

print("Restored Signal Data Format: {}".format(S_.shape))

pca = PCA(n_components=3)
H = pca.fit_transform(X)

models = [X, S, S_, H]
names = ['Detected Signals (First 3)',
         'Original Signal',
         'Restored by NMF',
         'Resotred by PCA']

fig, axes = plt.subplots(4, figsize=(8,4), gridspec_kw={'hspace': .5},
                         subplot_kw={'xticks': (), 'yticks': ()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')

# plt.figure(figsize=(6, 1))
# plt.plot(S, '-')
# plt.xlabel("Time")
# plt.ylabel("Signal")

plt.show()

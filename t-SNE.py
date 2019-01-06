import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

digits = load_digits()

pca = PCA(n_components=2)
pca.fit(digits.data)

tsne = TSNE(random_state=42)

digits_pca = pca.transform(digits.data)
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10,10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())

for i in range(len(digits_tsne)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             fontdict={'weight': 'bold', 'size':9})

plt.xlabel("1st main feature")
plt.ylabel("2nd main feature")

plt.show()

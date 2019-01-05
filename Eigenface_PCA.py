import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people/255

X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0
)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("1-KNeighbor Test Set Score: {:.2f}".format(knn.score(X_test, y_test)))

pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

knn.fit(X_train_pca, y_train)

print("1-KNeighbor Test Set Score: {:.2f}".format(knn.score(X_test_pca, y_test)))

# fig, axes = plt.subplots(2, 5, figsize=(15,8),
#                          subplot_kw={'xticks': (), 'yticks': ()})
#
# for target, image, ax in zip(people.target, people.images, axes.ravel()):
#     ax.imshow(image)
#     ax.set_title(people.target_names[target])
#
# print("people.images.shape: {}".format(people.images.shape))
# print("Num of Classes: {}".format(len(people.target_names)))

fig, axes = plt.subplots(3, 5, figsize=(15,12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel(0))):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("Main features {}".format((i + 1)))

plt.show()

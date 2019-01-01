from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import  export_graphviz
import graphviz
from IPython.display import display

def plot_feature_importatnces_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print("Train Set Score: {:.3f}".format(tree.score(X_train, y_train)))
print("Test Set Score: {:.3f}".format(tree.score(X_test, y_test)))

tree_pruned = DecisionTreeClassifier(max_depth=4, random_state=0)
tree_pruned.fit(X_train, y_train)

print("Train Set Score: {:.3f}".format(tree_pruned.score(X_train, y_train)))
print("Test Set Score: {:.3f}".format(tree_pruned.score(X_test, y_test)))

export_graphviz(tree_pruned, out_file="tree.dot", class_names=["Negative","Positive"],
                feature_names=cancer.feature_names,
                impurity=False, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

print("Feature Importance: \n{}".format(tree_pruned.feature_importances_))
plot_feature_importatnces_cancer(tree_pruned)
plt.show()

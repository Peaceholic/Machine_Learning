from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())

pipe.fit(cancer.data)

components = pipe.named_steps["pca"].components_

print("Pipeline steps:\n{}".format(pipe.steps))
print("components.shape: {}".format(components.shape))

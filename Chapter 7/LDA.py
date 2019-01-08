from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

reviews_train = load_files("C:/aclImdb/train/")
reviews_test = load_files("C:/aclImdb/test/")

text_train, y_train = reviews_train.data, reviews_train.target
text_test, y_test = reviews_test.data, reviews_test.target

vect = CountVectorizer(max_features=10000, max_df=.15)
X = vect.fit_transform(text_train)

lda = LatentDirichletAllocation(n_components=10, learning_method="batch", max_iter=25, random_state=0)

document_topics = lda.fit_transform(X)

sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())

mglearn.tools.print_topics(topics=range(10), feature_names=feature_names, sorting=sorting, topics_per_chunk=5, n_words=10)

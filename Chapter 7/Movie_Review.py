import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

reviews_train = load_files("C:/aclImdb/train/")
reviews_test = load_files("C:/aclImdb/test/")

text_train, y_train = reviews_train.data, reviews_train.target
text_test, y_test = reviews_test.data, reviews_test.target

vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
X_train = vect.transform(text_train)
X_test = vect.transform(text_test)

pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())

param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10],
              'tfidfvectorizer__ngram_range': [(1,1), (1,2), (1,3)]}

# print("text_train type: {}".format(type(text_train)))
# print("text_train length: {}".format(len(text_train)))
# print("text_test type: {}".format(type(text_test)))
# print("text_test length: {}".format(len(text_test)))

text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

# print("Num of samples (Train data): {}".format(np.bincount(y_train)))
# print("Num of samples (Test data): {}".format(np.bincount(y_test)))

scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)

# print("Cross Validation Mean Scores: {:.2f}".format(np.mean(scores)))
# print("Best Cross Validation Scores: {:.2f}".format(grid.best_score_))
# print("Best Parameter: {}".format(grid.best_params_))
# print("Test Score: {:.2f}".format(grid.score(X_test, y_test)))

vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]

X_train = vectorizer.transform(text_train)
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tidf = max_value.argsort()

feature_names = np.array(vectorizer.get_feature_names())

# print("Smallest tfidf feature:\n{}".format(feature_names[sorted_by_tidf[:20]]))
# print("Largest tfidf feature:\n{}".format(feature_names[sorted_by_tidf[-20:]]))

print("Best Cross Validation Score: {:.2f}".format(grid.best_score_))
print("Best Parameters: \n{}".format(grid.best_params_))

scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T

heatmap = mglearn.tools.heatmap(
    scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
    xticklabels=param_grid['logisticregression__C'],
    yticklabels=param_grid['tfidfvectorizer__ngram_range']
)

plt.colorbar(heatmap)
plt.show()

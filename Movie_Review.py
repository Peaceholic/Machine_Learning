from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_files

reviews_train = load_files("C:/aclImdb/train/")
reviews_test = load_files("C:/aclImdb/test/")

text_train, y_train = reviews_train.data, reviews_train.target
text_test, y_test = reviews_test.data, reviews_test.target

print("text_train type: {}".format(type(text_train)))
print("text_train length: {}".format(len(text_train)))
# print("text_train[6]:\n{}".format(text_train[6]))
print("text_test type: {}".format(type(text_test)))
print("text_test length: {}".format(len(text_test)))

text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

print("Num of samples (Train data): {}".format(np.bincount(y_train)))
print("Num of samples (Test data): {}".format(np.bincount(y_test)))

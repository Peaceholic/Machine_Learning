from IPython.display import display
import pandas as pd
import mglearn
import os

data = pd.read_csv(
    os.path.join(mglearn.datasets.DATA_PATH, "adult.data"),
    header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'martial-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income']
)

data = data[['age', 'workclass','education', 'gender', 'hours-per-week',
             'occupation', 'income']]

display(data.head())

print("Original Features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("After Features:\n", list(data_dummies.columns))

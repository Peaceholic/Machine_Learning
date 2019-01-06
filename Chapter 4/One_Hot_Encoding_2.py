from IPython.display import display
import pandas as pd

demo_df = pd.DataFrame({'Numerical Feature': [0, 1, 2, 1],
                        "Categorical Feature": ['Socks', 'Fox', 'Socks', 'Box']})

demo_df['Numerical Feature'] = demo_df['Numerical Feature'].astype(str)

display(pd.get_dummies(demo_df, columns=['Numerical Feature', 'Categorical Feature']))

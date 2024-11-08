import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])
print(data.shape) #(4, 5)

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

print(type(data))
print(data.dtypes)

print(np.__version__) #1.26.3
np.float = float
# pip install impyute
from impyute.imputation.cs import mice
data9 = mice(data.values,
             n = 10, 
             seed = 777)
print(data9)
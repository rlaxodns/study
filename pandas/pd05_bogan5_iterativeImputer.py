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

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# xgb도 가능함

imputer = IterativeImputer() # BaysianRidge 회귀모델
data1 = imputer.fit_transform(data)
print(data1)

imputer = IterativeImputer(estimator=DecisionTreeRegressor()) 
data2 = imputer.fit_transform(data)
print(data2)

imputer = IterativeImputer(estimator=RandomForestRegressor())
data3 = imputer.fit_transform(data)
print(data3)
from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4567)

Random_state = 8888
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
model1 = DecisionTreeRegressor(random_state=Random_state)
model2 = RandomForestRegressor(random_state=Random_state)
model3 = GradientBoostingRegressor(random_state=Random_state)
model4 = XGBRegressor(random_state = Random_state)

model = [model2]
model4.fit(x_train, y_train)

import matplotlib.pyplot as plt
from xgboost.plotting import plot_importance
plot_importance(model4)
plt.show()
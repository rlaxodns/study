from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_diabetes()
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

models = [model2]
model2.fit(x_train, y_train)

thre = np.sort(model2.feature_importances_)
from sklearn.feature_selection import SelectFromModel
for i in thre:
    select = SelectFromModel(
        model2, threshold= i , prefit = False
    )

    sel_x_train = select.transform(x_train)
    sel_x_test = select.transform(x_test)

    sel_model = model4 = XGBRegressor(random_state = Random_state)

    from sklearn.metrics import r2_score
    sel_model.fit(sel_x_train, y_train)
    sel_y_pre = sel_model.predict(sel_x_test)
    r2 = r2_score(y_test, sel_y_pre)

    print("thre", i, 'n=', sel_x_train.shape[1], 'r2', r2)
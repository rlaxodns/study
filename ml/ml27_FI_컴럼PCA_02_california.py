from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR


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

models = [model2]


print(Random_state)
for model in models:
    model.fit(x_train, y_train)
    print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
    print('acc', model.score(x_test, y_test))
    print(model.feature_importances_)


percentiles = np.percentile(model.feature_importances_, 25)

idx = []

for index, importance in enumerate(model.feature_importances_):
    if importance <= percentiles:
        idx.append(index)


print(idx)

x_train1 = np.delete(x_train, idx, axis = 1)
x_test1 = np.delete(x_test, idx, axis = 1)

x_train2 = x_train[:, idx]
x_test2 = x_test[:, idx]

# print(x_train1.shape, x_train2.shape) (16512, 6) (16512, 2)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x_train2 = pca.fit_transform(x_train2)
x_test2 = pca.transform(x_test2)

# print(x_train2.shape) (16512, 1)

x_train_con = np.concatenate((x_train1, x_train2), axis = 1)
# print(x_train_con.shape) (16512, 7)
x_test_con = np.concatenate((x_test1, x_test2), axis = 1)

model.fit(x_train_con, y_train)

print('========================', model.__class__.__name__, idx,'Random_state = ',Random_state, "======================")
print('acc', model.score(x_test_con, y_test))


# x_train1 = np.delete(x_train, idx, axis = 1)
# x_test1 = np.delete(x_test, idx, axis = 1)

# model = RandomForestRegressor(random_state=Random_state)
# model.fit(x_train1, y_train)

# print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
# print('acc', model.score(x_test1, y_test))
# print(np.percentile(model.feature_importances_, 25))
# print(model.feature_importances_)
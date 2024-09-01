from sklearn.datasets import fetch_california_housing, load_diabetes
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

x, y =load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4567)

model1 = RandomForestRegressor(random_state=777)

model1.fit(x_train, y_train)
per = np.percentile(model1.feature_importances_, 25)

idx = []
for index, values in enumerate(model1.feature_importances_):
    if values <= per:
        idx.append(index)

print(idx)

del_x_train = np.delete(x_train, idx, axis = 1)
del_x_test = np.delete(x_test, idx, axis = 1)

drop_x_train = x_train[:, idx]
drop_x_test = x_test[:, idx]

# print(del_x_train.shape, drop_x_train.shape) #(353, 7) (353, 3)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca_x_train = pca.fit_transform(drop_x_train)
pca_x_test = pca.transform(drop_x_test)

# print(del_x_train.shape, pca_x_train.shape) #(353, 7) (353, 1)
# print(del_x_test.shape, pca_x_test.shape) #(89, 7) (89, 1)
# exit()

con_x_train = np.concatenate((del_x_train, pca_x_train), axis = 1)
con_x_test = np.concatenate((del_x_test, pca_x_test), axis = 1)

model2 = XGBRegressor(random_state = 4123)

model2.fit(con_x_train, y_train)
print('========================', model2.__class__.__name__, idx,"======================")
print('acc', model2.score(con_x_test, y_test))

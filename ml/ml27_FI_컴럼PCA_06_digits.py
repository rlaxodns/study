from sklearn.datasets import load_diabetes, load_digits
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR


#1. 데이터
dataset = load_digits()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4567)

Random_state = 8888
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBRegressor
model1 = DecisionTreeRegressor(random_state=Random_state)
model2 = RandomForestClassifier(random_state=Random_state)
model3 = GradientBoostingClassifier(random_state=Random_state)
model4 = XGBRegressor(random_state = Random_state)

models = [model2]

model2.fit(x_train, y_train)

per = np.percentile(model2.feature_importances_, 25)

idx = []
for index, importance in enumerate(model2.feature_importances_):
    if importance <= per:
        idx.append(index)

del_x_train = np.delete(x_train, idx, axis = 1)
del_x_test = np.delete(x_test, idx, axis = 1)

d_x_train = x_train[:, idx]
d_x_test = x_test[:, idx]

from sklearn.decomposition import PCA
pca=PCA(n_components=1)
pca_x_train = pca.fit_transform(d_x_train)
pca_x_test = pca.transform(d_x_test)

#합치기
con_x_train = np.concatenate((del_x_train, pca_x_train), axis = 1)
con_x_test = np.concatenate((del_x_test, pca_x_test), axis = 1)

# 모델에 적용
model3.fit(con_x_train, y_train)
print("score", model3.score(con_x_test, y_test))
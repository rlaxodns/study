from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x)
# print(x.shape, y.shape) #(150, 4) (150,)
print(np.unique(y, return_counts = True))

Random_state = 8888
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.2, 
                                                    random_state=Random_state, 
                                                    stratify=y)


#2. 모델구성
model = RandomForestClassifier(random_state=Random_state)
model.fit(x_train, y_train)

per = np.percentile(model.feature_importances_, 25)
idx = []

for index, importance in enumerate(model.feature_importances_):
    if importance<=per:
        idx.append(index)

del_x_train = np.delete(x_train, idx, axis=1)
del_x_test = np.delete(x_test, idx, axis =1)

d_x_train = x_train[:, idx]
d_x_test = x_test[:, idx]

#pca적용
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca_x_train = pca.fit_transform(d_x_train)
pca_x_test = pca.transform(d_x_test)

#병합
con_x_train = np.concatenate((del_x_train, pca_x_train), axis = 1)
con_x_test = np.concatenate((del_x_train, pca_x_train), axis = 1)

#적용
model1 = XGBClassifier(Random_state=4566)
print(con_x_train.shape, y_train.shape)
model1.fit(con_x_train, y_train)
sel_y_pre = model1.predict(con_x_test)

from sklearn.metrics import accuracy_score
acc = model1.score(con_x_test, sel_y_pre)
print(acc)
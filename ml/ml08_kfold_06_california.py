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

print(x.shape)
print(y.shape)

kfold = KFold(n_splits=5,
              shuffle=True, 
              random_state=777)

# [실습]r2 059이상
#모델
model = SVR()

# 컴파일 및 훈련, # 예측 및 평가
score = cross_val_score(model, x, y, 
                        cv = kfold)

print(score)
print(np.mean(score))

"""[-0.02804881 -0.01709998 -0.04012168 -0.01175051 -0.02634697]
-0.02467358934159014"""
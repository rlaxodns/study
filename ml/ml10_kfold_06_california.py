from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.svm import SVR


#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    
                                                    )


kfold = KFold(n_splits=5,
              shuffle=True, 
              random_state=777)

# [실습]r2 059이상
#모델
model = SVR()

# 컴파일 및 훈련, # 예측 및 평가
score = cross_val_score(model, x_train, y_train, 
                        cv = kfold)
print(score)
print(np.mean(score))

y_pre = cross_val_predict(model, x_test, y_test, 
                          cv = kfold)

r2 = r2_score(y_test, y_pre)
print(r2)

"""[-0.02804881 -0.01709998 -0.04012168 -0.01175051 -0.02634697]
-0.02467358934159014


[-0.04090688 -0.03783129 -0.01491091 -0.04268083 -0.01822362]
-0.03091070611725528
-0.04755575926242228"""
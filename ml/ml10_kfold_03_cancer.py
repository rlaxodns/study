# 데이터에 있어서 두개(1, 0)를 나타내는 데이터에서 두개와 다른 데이터가 존재하는 경우에 있어서 
# 결과값에 지대한 영향을 미치게 되고 해당 문제를 찾기 위해 어려움이 발생할 수 있다

# 두개 중에 하나를 찾는 것, '이진분류'
# 두개 이상의 데이터를 분류 하는 것은 '다중분류'
# 지금까지와 달리 최종 결과값을 비교하여 맞는지 틀린지를 검토한다
# accuracy(정확성) 지표를 사용한다

import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC

#1 데이터
dataset = load_breast_cancer()
print(dataset.DESCR) 
print(dataset.feature_names)


x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    stratify=y
                                                    )

kfold = KFold(
    n_splits=5,
    shuffle=True, 
    random_state=777
)

#2. 모델
model = SVC()

#3. 컴파일 및 훈련
score = cross_val_score(model, x_train, y_train,
                        cv = kfold)
print("score", score, "평균score", np.mean(score))

y_pre = cross_val_predict(model, x_test, y_test, 
                          cv = kfold)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pre)
print(acc)
# score [0.90350877 0.92105263 0.9122807  0.90350877 0.9380531 ] 평균score 0.915680794907623
# score [0.94505495 0.89010989 0.87912088 0.89010989 0.9010989 ] 평균score 0.9010989010989011
# 0.9385964912280702

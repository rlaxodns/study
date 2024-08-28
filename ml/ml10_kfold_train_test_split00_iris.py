import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict # 교차 검증 점수에 대한 라이브러리
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#1. 데이터
x, y = load_iris(return_X_y = True)
# print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                     test_size=0.2,
                                                       random_state=777,
                                                         stratify=y
                                                         )

##스케일링 적용##
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
################

##kfold###############
n_split = 5
kfold = StratifiedKFold(n_splits = n_split, # KFold 클래스의 인스턴스를 불러온다
              shuffle=True,
              random_state=777)
######################

#2. 모델
model = SVC()

#3. 훈련
score = cross_val_score(model, x_train, y_train,    # 기준 점수 확인
                        cv = kfold)     # cross_val_score에 fit이 제공됨

print('ACC', score)
print('평균ACC', np.mean(score)) #평균ACC 0.95

y_predict = cross_val_predict(model, x_test, y_test,
                              cv = kfold)

print(y_predict)
print(y_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print(acc) #0.9333333333333333
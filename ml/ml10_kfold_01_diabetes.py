from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import numpy as np

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.svm import SVR

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) #(442, 10) (442,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=777,)



kfold = KFold(n_splits=5,
              shuffle=True,
              random_state=777)

#2. 모델
model = SVR()

#3. 컴파일 및 훈련
score = cross_val_score(model, x_train, y_train, 
                        cv = kfold)

#4. 에측 밒 평가
print("score", score, "평균score", np.mean(score))

y_pre = cross_val_predict(model, x_test, y_test, 
                          cv = kfold)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pre)
print(r2)

"""
score [0.11485129 0.14570299 0.19377434 0.14604525 0.00264189] 평균score 0.12060315425345121
score [0.08283497 0.16514883 0.10097781 0.05470424 0.15205747] 평균score 0.11114466130661294
0.05224992363363612
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score # 교차 검증 점수에 대한 라이브러리
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터
x, y = load_iris(return_X_y = True)
# print(x.shape)


n_split = 5
kfold = KFold(n_splits = n_split, # KFold 클래스의 인스턴스를 불러온다
              shuffle=True,
              random_state=777)

#2. 모델
model = SVC()

#3. 훈련
score = cross_val_score(model, x, y,
                        cv = kfold)     # cross_val_score에 fit이 제공됨

print('ACC', score)
print('평균ACC', np.mean(score))
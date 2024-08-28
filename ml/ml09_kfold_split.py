import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score # 교차 검증 점수에 대한 라이브러리
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터
data = load_iris()
# print(x.shape)

df = pd.DataFrame(data.data, columns = data.feature_names)
print(df)

n_split = 3
kfold = KFold(n_splits = n_split, # KFold 클래스의 인스턴스를 불러온다
              shuffle=False,
            #   random_state=777
              )

for train_index, val_index in kfold.split(df):
    print("===================================")
    print(train_index, '\n', val_index)
    print('훈련데이터의 수', len(train_index), '\n',
          '검즘데이터의 수', len(val_index))
# 훈련데이터의 수 100
# 검즘데이터의 수 50 
# https://dacon.io/competitions/official/236068/overview/description

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\train.csv", index_col = 0)
test = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\test.csv", index_col=0)
submission = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\sample_submission.csv", index_col = 0)


x = train.drop(['Outcome'], axis=1)
y = train['Outcome']

kfold = KFold(n_splits=5, 
              shuffle=True, 
              random_state=777)

#2. 모델 구성
model = SVC()

#3. 훈련 및 에측
score = cross_val_score(model, x, y, 
                        cv=kfold)

print("score", score, "평균score", np.mean(score))
# score [0.71755725 0.79389313 0.76153846 0.78461538 0.73846154] 평균score 0.7592131532589548
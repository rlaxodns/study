
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# 모델
data = load_wine()
x = data.data #(178, 13)
y = data.target #(178, 3)

# y = pd.get_dummies(y)

# kfold = KFold(
#     n_splits=5, 
#     shuffle=True,
#     random_state=777
# )

kfold = StratifiedKFold(
    n_splits=5, 
    shuffle=True,
    random_state=777
)

#모델 구성
model = SVC()

#훈련 및 예측
score = cross_val_score(
    model, x, y, 
    cv=kfold
)

print("score", score, "평균score", np.mean(score))
# score [0.71755725 0.79389313 0.76153846 0.78461538 0.73846154] 평균score 0.7592131532589548
# score [0.75       0.61111111 0.69444444 0.71428571 0.6       ] 평균score 0.673968253968254
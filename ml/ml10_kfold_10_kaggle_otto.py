import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import random as rn
rn.seed(6265)
tf.random.set_seed(6265)
np.random.seed(6265)

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# 데이터 구성
path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sub = pd.read_csv(path + "sampleSubmission.csv", index_col=0, )

le = LabelEncoder()
train["target"] = le.fit_transform(train["target"])

x = train.drop(['target'], axis=1)
y = train['target']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    stratify=y
                                                    )

kfold = StratifiedKFold(
    n_splits=7,
    shuffle = True,
    random_state=777,
    
)

# 모델
model = xgb.XGBClassifier(
    n_estimators = 100,
    learning_rate = 0.7,
    max_depth = 3,
    random_state = 777
)

# 평가 및 예측
score = cross_val_score(model, x_train, y_train, 
                        cv = kfold)

print("score", score, "평균score", np.mean(score))

y_pre = cross_val_predict(model, x_test, y_test, 
                          cv = kfold)
acc = accuracy_score(y_test, y_pre)
print(acc)
# score [0.79977376 0.79445701 0.79717195 0.79660633 0.80316742 0.80552099
#  0.79907229] 평균score 0.799395678624431

# score [0.80554299 0.79920814 0.79977376 0.80079186 0.79524887 0.79635705
#  0.79760154] 평균score 0.799217743351529
# 0.89985
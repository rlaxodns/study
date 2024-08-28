# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# 데이터 구성
path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"
train = pd.read_csv(path+"train.csv", index_col=0)
test = pd.read_csv(path+"test.csv", index_col=0)
sub = pd.read_csv(path+"sample_submission.csv", index_col = 0)

x = train.drop(['target'], axis = 1)
y = train['target']

kfold = StratifiedKFold(
    n_splits=7,
    shuffle = True,
    random_state=777
)


#2. 모델구성 
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.5,
    max_depth=16,
    random_state=7777,
    use_label_encoder=False,
    eval_metric='mlogloss', 
    gamma = 3,
    
)

# 평가 및 예측
score = cross_val_score(model, x, y, 
                        cv = kfold)

print("score", score, "평균score", np.mean(score))

# score [0.90560689 0.90154697 0.90599188 0.90073851 0.90259354 0.90462357
#  0.90199853] 평균score 0.9032999837710711

# score [0.90242195 0.90441691 0.90466191 0.90353855 0.90339855 0.90245354
#  0.90339855] 평균score 0.9034699945460682
# train['종가'] = train['종가'].str.replace(',', '')
# https://www.kaggle.com/c/playground-series-s4e1/data
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\train.csv", index_col=[0, 1, 2])
test = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\test.csv", index_col= [0, 1, 2])
submission = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\sample_submission.csv", index_col=[0])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train['Geography'] = le.fit_transform(train['Geography'])
train['Gender'] = le.fit_transform(train['Gender'])

test['Geography'] = le.fit_transform(test['Geography'])
test['Gender'] = le.fit_transform(test['Gender'])

x = train.drop(['Exited'], axis = 1)
y = train['Exited']

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])

kfold = StratifiedKFold(n_splits=7,
              shuffle=True, 
              random_state=777)

#2. 모델 구성
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
)

#3. 훈련 및 예측
score = cross_val_score(model, x, y, 
                        cv = kfold)

print("score", score, "평균score", np.mean(score))
# score [0.90350877 0.92105263 0.9122807  0.90350877 0.9380531 ] 평균score 0.915680794907623
# score [0.86172965 0.86919455 0.86579572 0.86702579 0.86689854 0.86558364
#  0.86371734] 평균score 0.8657064634482659
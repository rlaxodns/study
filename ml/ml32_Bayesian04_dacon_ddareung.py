# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd #csv파일 사용시 데이터의 인덱스와 컬럼을 분리시 사용
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')


#1. data

path = "./_data/dacon/따릉이/" # 상대경로
train_csv = pd.read_csv(path + "train.csv", index_col=0) 
test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv", index_col=0) 
train_csv = train_csv.dropna() #데이터 내의 na값을 제외한다.
test_csv = test_csv.fillna(test_csv.mean())  # .fillna  : 결측치를 채우는 함수
                                             # .mean() 각 컬럼의 평균값을 채워진다.

x = train_csv.drop(['count'], axis=1)  # axis=0은 행의 [---]이름의 행 삭제, 
# axis=1는 열의 [---]이름의 열 삭제


y = train_csv['count'] #y는 카운트 열만 지정
print(y.shape) # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        test_size=0.2, shuffle=True, random_state=4345)
# print(x_train.shape) # (929, 9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = mas.fit_transform(x_train)
x_test = mas.transform(x_test)


#2. 모델
bayesian_params = {
    'learning_rate':(0.001, 0.1),
    'max_depth':(3, 10),
    'num_leaves':(24, 40),
    'min_child_samples':(10, 200),
    'min_child_weight':(1, 50),
    'subsample':(0.5, 1),
    'colsample_bytree':(0.5, 1),
    'max_bin':(9, 500),
    'reg_lambda':(-0.001, 10),
    'reg_alpha':(0.01, 50)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample,colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'num_leaves': int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        'min_child_weight':int(round(min_child_weight)),
        'subsample':max(min(subsample, 1), 0),
        'colsample_bytree':colsample_bytree,
        'max_bin':max(int(round(max_bin)), 10),
        'reg_lambda':max(reg_lambda, 0),
        'reg_alpha':reg_alpha
    }

    model = XGBRegressor(**params,n_jobs = -1)
    model.fit(x_train, y_train,
              eval_set= [(x_test, y_test)],
            #   eval_metrics = 'logloss',
              verbose = 0)
    
    y_predict = model.predict(x_test)
    result = r2_score(y_test, y_predict)
    
    return result

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
st = time.time()
bay.maximize(init_points=5,
             n_iter=n_iter)
et = time.time()

print(bay.max)
print(n_iter, "걸린 시간", round(et-st, 2))
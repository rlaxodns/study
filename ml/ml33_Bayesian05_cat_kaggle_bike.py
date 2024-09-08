
# x의 공백을 예측하여, 해당 예측치를 가지고 y의 값을 예측

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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



#1. 데이터 준비

train_csv = pd.read_csv('C:\\ai5\\_data\\kaggle\\bike-sharing-demand\\train.csv', index_col=0)
test2_csv = pd.read_csv('C:\\ai5\\_data\\kaggle\\bike-sharing-demand\\test2.csv', index_col=0)
sample_csv = pd.read_csv('C:\\ai5\_data\\kaggle\\bike-sharing-demand\\sampleSubmission.csv', index_col=0)

print(train_csv.shape)
print(test2_csv.shape)

import matplotlib.pyplot as plt
# train_csv.boxplot() 시리즈에서는 이건 안된다.
# train_csv.plot.box()

##1. x와 y의 분리
x = train_csv.drop(['count'], axis=1)
y = train_csv[['count']]

print(x) #[10886 rows x 10 columns]
print(y) #[10886 rows x 1 columns]


##2. 훈련, 검증 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
# print(x_train.shape, y_train.shape)


from catboost import CatBoostRegressor
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
        # 'num_leaves': int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        # 'min_child_weight':int(round(min_child_weight)),
        # 'subsample':max(min(subsample, 1), 0),
        # 'colsample_bytree':colsample_bytree,
        'max_bin':max(int(round(max_bin)), 10),
        'reg_lambda':max(reg_lambda, 0),
        # 'reg_alpha':reg_alpha,
    }

    model = CatBoostRegressor(**params,
                               task_type='GPU',         # GPU사용 기본은CPU
                                devices='0',
                                early_stopping_rounds=100, #조기종료 기본값 None
                                verbose=10,  )
    model.fit(x_train, y_train,
              eval_set= (x_test, y_test),
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

"""
{'target': 0.999733593588244, 'params': {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_bin': 451.4513240537555, 'max_depth': 7.021525088553034, 'min_child_samples': 151.55218332951597, 'min_child_weight': 35.90729493236813, 'num_leaves': 24.0, 'reg_alpha': 33.058397689800124, 'reg_lambda': 0.5205626552203956, 'subsample': 0.5}}
100 걸린 시간 91.08
"""
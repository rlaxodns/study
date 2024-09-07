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
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

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
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

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

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    stratify=y
                                                    )

#2. 모델
from catboost import CatBoostClassifier
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

    model = CatBoostClassifier(**params,
                               task_type='GPU',         # GPU사용 기본은CPU
                                devices='0',
                                early_stopping_rounds=100, #조기종료 기본값 None
                                verbose=10,  )
    model.fit(x_train, y_train,
              eval_set= (x_test, y_test),
            #   eval_metrics = 'logloss',
              verbose = 0)
    
    y_predict = model.predict(x_test)
    result = accuracy_score(y_test, y_predict)
    
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
{'target': 0.8649377404792923, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_bin': 293.81884685198554, 'max_depth': 10.0, 'min_child_samples': 54.13986136120862, 'min_child_weight': 23.892247504903533, 'num_leaves': 24.0, 'reg_alpha': 47.46348005989001, 'reg_lambda': -0.001, 'subsample': 0.5}}
100 걸린 시간 121.17"""
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

#1. 데이터
x, y = load_diabetes(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=777,)

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test  = mms.transform(x_test)


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

# {'target': 0.5340608931193684, 'params': {'colsample_bytree': 0.7888545577843505, 'learning_rate': 0.1, 'max_bin': 100.2463666260029, 'max_depth': 8.5351458827384, 'min_child_samples': 121.42076824320898, 'min_child_weight': 49.914424243733436, 'num_leaves': 32.246525820816636, 'reg_alpha': 33.98709425782796, 'reg_lambda': 7.4109445693120675, 'subsample': 0.6773681227370609}}
# 100 걸린 시간 12.78
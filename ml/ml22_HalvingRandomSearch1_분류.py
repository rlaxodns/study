# 격자탐색_cross_validation

import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

from sklearn.model_selection import RandomizedSearchCV 

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    stratify=y
                                                    ) 

# print(x_train.shape, y_train.shape) #(1437, 64) (1437,)
# print(np.unique(y, return_counts = True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
# exit()

kfold = StratifiedKFold(
    n_splits=5, 
    shuffle=True, 
    random_state= 777
)

parameter = [
    {'learning_rate' :[0.01, 0.05, 0.1, 0.2, 0.5], 'max_depth':[3,4,5,6,8]}, 
    {'learning_rate' :[0.01, 0.05, 0.1, 0.2, 0.3], 'subsample':[0.6, 0.7,0.8, 0.9, 1.0]},
    {'learning_rate' :[0.01, 0.05, 0.1, 0.2, 0.3], 'colsample_bytree':[0.6,0.7,0.8,0.9,1.0]},
    {'learning_rate' :[0.01, 0.05, 0.1, 0.2, 0.3], 'gamma':[0,0.1,0.2, 0.3,0.5,1.0]}     
    ]  #

#2. 모델

model = HalvingRandomSearchCV(XGBClassifier(
    # tree_method = 'gpu_hist', 과거 방식
                    tree_method = 'hist',
                    device = 'cuda', 
                    n_estimators = 50,
                    ),
parameter, 
cv=kfold, 
refit = True,
verbose=1,  #xgboost는 verbose 2or3으로 확인
# n_jobs=-1,  # cpu의 모든 코어를 활용함
# n_iter= 0,  # random_search의 횟수 조절
random_state=777,
min_resources=30, 
# max_resources=1100,
# aggressive_elimination=True, # True의 경우 제거의 폭을 넓힌다. 제거 폭은 factor+1만큼 증가 
factor = 3 # factor는 parameter의 수는 factor만큼 줄여서 최적의 파라미터를 찾은 후, 
# data를 다시 늘려서 최적의 파라미터를 활용해서 최적의 파라미터를 다시 파악하고, 
# factor만큼 계속 iter반복, (facor = 3.5 등 소수점도 가능)
) 



st = time.time()


model.fit(x_train, y_train,) #eval_set은 얼리스탑핑 및 validation을 활용한다


et = time.time()

# Fitting 5 folds for each of 54 candidates, totalling 270 fits
print('최적의 매개변수', model.best_estimator_) 
# 최적의 매개변수 SVC(C=1, kernel='linear')

print('최적의 파라미터', model.best_params_)
# 최적의 파라미터 {'C': 1, 'degree': 3, 'kernel': 'linear'}

print('best_score', model.best_score_) # train을 통한 최고의 점수를 의미하기 때문에 과적합된 점수
# best_score 0.9916666666666666

print('model.score', model.score(x_test, y_test))
# model.score 0.9333333333333333

y_pre = model.predict(x_test)
print("acc", accuracy_score(y_test, y_pre))
# acc 0.9333333333333333

y_pre_best = model.best_estimator_.predict(x_test)
print("최적의 튜닝acc", accuracy_score(y_test, y_pre_best) )
# 최적의 튜닝acc 0.9333333333333333

print("걸린 시간", round(et-st, 2))
# 걸린 시간 1.82





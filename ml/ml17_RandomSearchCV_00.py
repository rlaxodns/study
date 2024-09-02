# 격자탐색_cross_validation

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

from sklearn.model_selection import RandomizedSearchCV

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    stratify=y
                                                    ) 

kfold = StratifiedKFold(
    n_splits=5, 
    shuffle=True, 
    random_state= 777
)

parameter = [
    {"C":[1, 10, 100, 1000], "kernel":['linear', 'sigmoid'], 'degree':[3,4,5],}, # 24번
    {"C":[1, 10, 100], "kernel": ['rbf'], 'gamma':[0.001, 0.0001]}, # 6번
    {"C":[1,10, 100, 1000], "kernel":['sigmoid'], 'gamma':[0.01, 0.001, 0.0001], 'degree':[3, 4]} # 24번
    ] # 전체 횟수는 54회  // 딕셔너리 내에서는 곱으로 횟수가 정해지고 각각 딕셔너리들의 연산 횟수는 단순 합

#2. 모델

model = RandomizedSearchCV(SVC(),
                    parameter, 
                    cv=kfold, 
                    refit = True,
                    verbose=1,
                    n_jobs=-1,  # cpu의 모든 코어를 활용함
                    n_iter= 10,  # random_search의 횟수 조절
                    random_state=777
                    )

st = time.time()
model.fit(x_train, y_train)
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





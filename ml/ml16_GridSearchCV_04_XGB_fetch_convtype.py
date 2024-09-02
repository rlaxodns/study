from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import xgboost as xgb

# 데이터 구성
data = fetch_covtype()
x = data.data
y = data.target
y = y - 1


print(np.unique(y))
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

parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'tree_method' : ['gpu_hist']}, #12
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #4
]  # 전체 횟수는 54회  // 딕셔너리 내에서는 곱으로 횟수가 정해지고 각각 딕셔너리들의 연산 횟수는 단순 합

#2. 모델
model = GridSearchCV(xgb.XGBClassifier(),
                    parameters, 
                    cv=kfold, 
                    refit = True,
                    verbose=1,
                    n_jobs=-1  # cpu의 모든 코어를 활용함
                    )

st = time.time()
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)
et = time.time()

print('최적의 매개변수', model.best_estimator_) 
# 최적의 매개변수 XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=None, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=12, 
# max_leaves=None,
#               min_child_weight=None, min_samples_leaf=3, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=500,
#               n_jobs=-1, num_parallel_tree=None, 
# ...)

print('최적의 파라미터', model.best_params_)
# 최적의 파라미터 {'max_depth': 12, 'min_samples_leaf': 3, 'n_estimators': 500, 'n_jobs': -1, 'tree_method': 'gpu_hist'}

print('best_score', model.best_score_) # train을 통한 최고의 점수를 의미하기 때문에 과적합된 점수
# best_score 0.9700371552403297

print('model.score', model.score(x_test, y_test))
# model.score 0.9725738578177844

y_pre = model.predict(x_test)
print("acc", accuracy_score(y_test, y_pre))
# acc 0.9725738578177844

y_pre_best = model.best_estimator_.predict(x_test)
print("최적의 튜닝acc", accuracy_score(y_test, y_pre_best) )
# 최적의 튜닝acc 0.9725738578177844

print("걸린 시간", round(et-st, 2))
# 걸린 시간 2483.2

import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)
print(pd.DataFrame(model.cv_results_).sort_values("rank_test_score", ascending=True))
print(pd.DataFrame(model.cv_results_).columns)
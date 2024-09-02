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
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import time

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

parameters = [
    {'n_jobs' : [-1], 'n_estimators' : [100, 500], 'max_depth' : [6, 10 ,12],
     'min_samples_leaf' : [3, 10], 'tree_method' : ['gpu_hist']}, #12
    {'n_jobs' : [-1], 'max_depth' : [6, 8,  10 ,12],
     'min_samples_leaf' : [3, 5, 7, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7, 10], 
     'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #16
    {'n_jobs' : [-1], 'min_samples_leaf' : [2, 3, 5, 10], 'tree_method' : ['gpu_hist']}, #4
] 

kfold = StratifiedKFold(
    n_splits=7,
    shuffle = True,
    random_state=777,
    
)

#2. 모델
model = RandomizedSearchCV(xgb.XGBClassifier(),
                    parameters, 
                    cv=kfold, 
                    refit = True,
                    verbose=1,
                    n_jobs=-1 ,  # cpu의 모든 코어를 활용함
                    n_iter= 10,  # random_search의 횟수 조절
                    random_state=777
                    )
st = time.time()
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = True)
et = time.time()

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

import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)
print(pd.DataFrame(model.cv_results_).sort_values("rank_test_score", ascending=True))
print(pd.DataFrame(model.cv_results_).columns)
from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    
                                                    )

##kfold###############
n_split = 5
kfold = KFold(n_splits = n_split, # KFold 클래스의 인스턴스를 불러온다
              shuffle=True,
              random_state=777)
######################

#2. 모델
from sklearn.utils import all_estimators
import sklearn as sk

all = all_estimators(type_filter='regressor')
print('all', all)
print('all', len(all)) # all 41
print(sk.__version__)  # 0.24.2

for name, model in all:
    try:
        #2. 모델
        model = model()
        
        #3. 훈련
        model.fit(x_train, y_train)

        #4. 평가
        acc = model.score(x_test, y_test)
        print(name, "의 정답률", acc)
    
    except: 
        print(name, 'error')
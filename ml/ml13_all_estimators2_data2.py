from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston, load_diabetes
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
cal = fetch_california_housing(return_X_y = True)
bos = load_boston(return_X_y = True)
dia = load_diabetes(return_X_y = True)

data = [cal, bos, dia]
data_name = ['캘리포니아', '보스톤', '디아벳츠']

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

import time
st = time.time()


aaa =[]
for index, value in enumerate(data):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    
                                                    )
    for name, model in all:
        try:
            #2. 모델
            model = model()
            
            
            #3. 훈련
            model.fit(x_train, y_train)
            score = cross_val_score(model, x_train, y_train,
                                    cv = kfold)
            y_pre = cross_val_predict(model, x_test, y_test,
                                    cv = kfold)
            #4. 평가
            cva = accuracy_score(y_test, y_pre).mean()
            
            print("acc", score)
            print("평균", np.mean(score))
            print('cross_val_acc', cva)
        except: 
            print(name, 'error') 
        
    aaa.append((data_name, name, cva))
if aaa:
    best_result = max(aaa, key=lambda x: x[2])  # cva 기준으로 가장 높은 값 찾기
    print(f"Best Result: Data Name = {best_result[0]}, Model = {best_result[1]}, Cross-Val Accuracy = {best_result[2]}")

et = time.time()
print(et - st)
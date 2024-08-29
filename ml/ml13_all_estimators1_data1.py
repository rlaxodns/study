import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict # 교차 검증 점수에 대한 라이브러리
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score


#1. 데이터
iris = load_iris(return_X_y = True)
cancer = load_breast_cancer(return_X_y = True)
wine = load_wine(return_X_y = True)
digit = load_digits(return_X_y = True)

data = [iris, cancer, wine, digit]
data_name = ['아이리스', '캔서', '와인', '디지트']



##kfold###############
n_split = 5
kfold = KFold(n_splits = n_split, # KFold 클래스의 인스턴스를 불러온다
              shuffle=True,
              random_state=777)
######################

#2. 모델
from sklearn.utils import all_estimators
import sklearn as sk

all = all_estimators(type_filter='classifier')
# print('all', all)
print('all', len(all)) # all 41
# print(sk.__version__)  # 0.24.2

import time
st = time.time()

aaa = []
for index, value in enumerate(data):
    x, y = value

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                     test_size=0.2,
                                                       random_state=777,
                                                         stratify=y
                                                         )

    ##스케일링 적용##
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    ################

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
            cva = r2_score(y_test, y_pre)

            print("=======", data_name[index], name,'============')
            print("acc", score)
            print("평균", np.mean(score))
            print('cross_val_acc', cva)
        
        except: 
            print("=======", data_name[index], name,'============')
            print(name, 'error')     

    aaa.append((data_name, name, cva))
if aaa:
    best_result = max(aaa, key=lambda x: x[2])  # cva 기준으로 가장 높은 값 찾기
    print(f"Best Result: Data Name = {best_result[0]}, Model = {best_result[1]}, Cross-Val Accuracy = {best_result[2]}")

et = time.time()
print(et - st)
    # Best Result: Data Name = ['아이리스', '캔서', '와인', '디지트'], Model = CalibratedClassifierCV, Cross-Val Accuracy = 1.0
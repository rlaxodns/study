import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict # 교차 검증 점수에 대한 라이브러리
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
 
#1. 데이터
x, y = load_iris(return_X_y = True)
# print(x.shape)

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

#2. 모델
from sklearn.utils import all_estimators
import sklearn as sk

all = all_estimators(type_filter='classifier')
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
    
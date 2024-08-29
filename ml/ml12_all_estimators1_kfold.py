import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict # 교차 검증 점수에 대한 라이브러리
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score

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
print('all', all)
print('all', len(all)) # all 41
print(sk.__version__)  # 0.24.2

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
        
        print("acc", score)
        print("평균", np.mean(score))
        print('cross_val_acc', cva)
    except: 
        print("=======", name,'============')
        print(name, 'error')     
"""
acc [0.95833333 1.         0.95833333 0.91666667 0.91666667]
평균 0.95
cross_val_acc 0.85
acc [0.875      1.         0.95833333 1.         0.875     ]
평균 0.9416666666666668
cross_val_acc 0.9
acc [0.70833333 0.79166667 0.79166667 0.91666667 0.625     ]
평균 0.7666666666666666
cross_val_acc 0.55
acc [0.875      0.91666667 0.875      0.91666667 0.79166667]
평균 0.875
cross_val_acc 0.5
CategoricalNB error
ClassifierChain error
ComplementNB error
acc [0.91666667 1.         0.95833333 1.         0.83333333]
평균 0.9416666666666667
cross_val_acc 0.85
acc [0.20833333 0.25       0.29166667 0.16666667 0.125     ]
평균 0.20833333333333331
cross_val_acc -0.7
acc [0.875      1.         0.95833333 0.91666667 0.875     ]
평균 0.925
cross_val_acc 0.7
acc [0.875 1.    1.    1.    0.875]
평균 0.95
cross_val_acc 0.95
acc [0.875      1.         1.         1.         0.91666667]
평균 0.9583333333333334
cross_val_acc 0.85
acc [0.83333333 1.         1.         1.         0.83333333]
평균 0.9333333333333333
cross_val_acc 0.9
acc [0.875 1.    1.    1.    0.875]
평균 0.95
cross_val_acc 0.9
acc [0.875      1.         0.95833333 1.         0.875     ]
평균 0.9416666666666668
cross_val_acc -0.7
acc [0.875      1.         0.95833333 1.         0.875     ]
평균 0.9416666666666668
cross_val_acc 0.75
acc [0.875 1.    1.    1.    0.875]
평균 0.95
cross_val_acc 0.75
acc [0.875 1.    1.    1.    0.875]
평균 0.95
cross_val_acc 0.75
acc [0.91666667 1.         1.         1.         0.95833333]
평균 0.975
cross_val_acc 0.85
acc [0.875      0.95833333 0.91666667 0.95833333 0.83333333]
평균 0.9083333333333334
cross_val_acc 0.7
acc [0.875 1.    1.    1.    0.875]
평균 0.95
cross_val_acc 0.85
acc [0.91666667 1.         1.         1.         0.875     ]
평균 0.9583333333333333
cross_val_acc 0.85
acc [0.875      0.95833333 1.         1.         0.79166667]
평균 0.925
cross_val_acc 0.85
MultiOutputClassifier error
MultinomialNB error
acc [0.66666667 0.95833333 0.91666667 0.875      0.79166667]
평균 0.8416666666666666
cross_val_acc 0.85
acc [0.83333333 1.         1.         1.         0.83333333]
평균 0.9333333333333333
cross_val_acc 0.8
OneVsOneClassifier error
OneVsRestClassifier error
OutputCodeClassifier error
acc [0.83333333 0.83333333 0.91666667 0.91666667 0.79166667]
평균 0.8583333333333334
cross_val_acc 0.6
acc [0.79166667 0.95833333 0.91666667 0.91666667 0.95833333]
평균 0.9083333333333332
cross_val_acc 0.75
acc [1.         1.         1.         1.         0.83333333]
평균 0.9666666666666666
cross_val_acc 0.75
RadiusNeighborsClassifier error
acc [0.875 1.    1.    1.    0.875]
평균 0.95
cross_val_acc 0.95
acc [0.75       0.875      0.875      0.875      0.83333333]
평균 0.8416666666666666
cross_val_acc 0.6
acc [0.79166667 0.875      0.875      0.875      0.83333333]
평균 0.85
cross_val_acc 0.55
acc [0.83333333 0.95833333 0.91666667 0.95833333 0.79166667]
평균 0.8916666666666668
cross_val_acc 0.5
acc [0.83333333 1.         1.         1.         0.83333333]
평균 0.9333333333333333
cross_val_acc 0.8
StackingClassifier error
VotingClassifier error
"""

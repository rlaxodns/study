from sklearn.datasets import fetch_california_housing
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
        score = cross_val_score(model, x_train, y_train,
                                cv = kfold)
        y_pre = cross_val_predict(model, x_test, y_test,
                                cv = kfold)
        #4. 평가
        cva = accuracy_score(y_test, y_pre)
        
        print("acc", score)
        print("평균", np.mean(score))
        print('cross_val_acc', cva)
    except: 
        print(name, 'error')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

"""
ARDRegression error
AdaBoostRegressor error
BaggingRegressor error
BayesianRidge error
CCA error
DecisionTreeRegressor error
DummyRegressor error
ElasticNet error
ElasticNetCV error
ExtraTreeRegressor error
ExtraTreesRegressor error
GammaRegressor error
GaussianProcessRegressor error
GradientBoostingRegressor error
HistGradientBoostingRegressor error
HuberRegressor error
IsotonicRegression error
KNeighborsRegressor error
KernelRidge error
Lars error
LarsCV error
Lasso error
LassoCV error
LassoLars error
LassoLarsCV error
LassoLarsIC error
LinearRegression error
LinearSVR error
MLPRegressor error
MultiOutputRegressor error
MultiTaskElasticNet error
MultiTaskElasticNetCV error
MultiTaskLasso error
MultiTaskLassoCV error
NuSVR error
OrthogonalMatchingPursuit error
OrthogonalMatchingPursuitCV error
PLSCanonical error
PLSRegression error
PassiveAggressiveRegressor error
PoissonRegressor error
RANSACRegressor error
RadiusNeighborsRegressor error
RandomForestRegressor error
RegressorChain error
Ridge error
RidgeCV error
SGDRegressor error
SVR error
StackingRegressor error
TheilSenRegressor error
TransformedTargetRegressor error
TweedieRegressor error"""

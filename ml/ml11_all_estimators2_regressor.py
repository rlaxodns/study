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

        """
        ARDRegression 의 정답률 0.572584215560146
        AdaBoostRegressor 의 정답률 0.4162972300885268
        BaggingRegressor 의 정답률 0.7812933645634604
        BayesianRidge 의 정답률 0.5829428921783684
        CCA 의 정답률 0.5482813514146694
        DecisionTreeRegressor 의 정답률 0.6192618154889691
        DummyRegressor 의 정답률 -1.381423168855811e-05
        ElasticNet 의 정답률 0.4110719481514774
        ElasticNetCV 의 정답률 0.5710297751379405
        ExtraTreeRegressor 의 정답률 0.5749956052889106
        ExtraTreesRegressor 의 정답률 0.8146544344936916
        GammaRegressor 의 정답률 -1.3946106516726786e-05
        GaussianProcessRegressor 의 정답률 -2.881690259196648
        GradientBoostingRegressor 의 정답률 0.779706452027026
        HistGradientBoostingRegressor 의 정답률 0.8314719923668268
        HuberRegressor 의 정답률 0.4593468705763083
        IsotonicRegression error
        KNeighborsRegressor 의 정답률 0.1423062168597895
        KernelRidge 의 정답률 0.5275090142697165
        Lars 의 정답률 0.5827784121245312
        LarsCV 의 정답률 0.5850979894080226
        Lasso 의 정답률 0.2827758158775012
        LassoCV 의 정답률 0.5753394061274506
        LassoLars 의 정답률 -1.381423168855811e-05
        LassoLarsCV 의 정답률 0.5827784121245312
        LassoLarsIC 의 정답률 0.5829896548343376
        LinearRegression 의 정답률 0.582778412124533
        LinearSVR 의 정답률 0.38522460639598144
        MLPRegressor 의 정답률 -0.4314645354059119
        MultiOutputRegressor error
        MultiTaskElasticNet error
        MultiTaskElasticNetCV error
        MultiTaskLasso error
        MultiTaskLassoCV error
        NuSVR 의 정답률 0.0070166773331958865
        OrthogonalMatchingPursuit 의 정답률 0.4493726103065231
        OrthogonalMatchingPursuitCV 의 정답률 0.5799682448855283
        PLSCanonical 의 정답률 0.34657166409209683
        PLSRegression 의 정답률 0.5021025788995132
        PassiveAggressiveRegressor 의 정답률 0.4469863691960152
        PoissonRegressor 의 정답률 -1.469682356480817e-05
        RANSACRegressor 의 정답률 0.23740207466741237
        RadiusNeighborsRegressor error
        RandomForestRegressor 의 정답률 0.8048527519832777
        RegressorChain error
        Ridge 의 정답률 0.5828225428195147
        RidgeCV 의 정답률 0.5832021925729557
        SGDRegressor 의 정답률 -3.451603463368467e+29
        SVR 의 정답률 -0.02804881267694581
        StackingRegressor error
        TheilSenRegressor 의 정답률 -1.0148025914682108
        TransformedTargetRegressor 의 정답률 0.582778412124533
        TweedieRegressor 의 정답률 0.48027430280936534
        VotingRegressor error
        """
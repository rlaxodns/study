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

        """
        AdaBoostClassifier 의 정답률 0.9666666666666667
BaggingClassifier 의 정답률 0.9333333333333333
BernoulliNB 의 정답률 0.7333333333333333
CalibratedClassifierCV 의 정답률 0.9666666666666667
CategoricalNB error
ClassifierChain error
ComplementNB error
DecisionTreeClassifier 의 정답률 0.9333333333333333
DummyClassifier 의 정답률 0.3333333333333333
ExtraTreeClassifier 의 정답률 0.9666666666666667
ExtraTreesClassifier 의 정답률 0.9333333333333333
GaussianNB 의 정답률 0.9666666666666667
GaussianProcessClassifier 의 정답률 0.9333333333333333
GradientBoostingClassifier 의 정답률 0.9333333333333333
HistGradientBoostingClassifier 의 정답률 0.9333333333333333
KNeighborsClassifier 의 정답률 0.9333333333333333
LabelPropagation 의 정답률 0.9333333333333333
LabelSpreading 의 정답률 0.9333333333333333
LinearDiscriminantAnalysis 의 정답률 0.9333333333333333
LinearSVC 의 정답률 0.9333333333333333
LogisticRegression 의 정답률 0.9333333333333333
LogisticRegressionCV 의 정답률 0.9333333333333333
MLPClassifier 의 정답률 0.9333333333333333
MultiOutputClassifier error
MultinomialNB error
NearestCentroid 의 정답률 0.8666666666666667
NuSVC 의 정답률 0.9666666666666667
OneVsOneClassifier error
OneVsRestClassifier error
OutputCodeClassifier error
PassiveAggressiveClassifier 의 정답률 0.8666666666666667
Perceptron 의 정답률 0.8666666666666667
QuadraticDiscriminantAnalysis 의 정답률 0.9333333333333333
RadiusNeighborsClassifier 의 정답률 1.0
RandomForestClassifier 의 정답률 0.9333333333333333
RidgeClassifier 의 정답률 0.8
RidgeClassifierCV 의 정답률 0.8
SGDClassifier 의 정답률 0.9333333333333333
SVC 의 정답률 0.9666666666666667
StackingClassifier error
VotingClassifier error"""
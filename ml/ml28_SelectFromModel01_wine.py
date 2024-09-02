from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target
print(x)
# print(x.shape, y.shape) #(150, 4) (150,)
print(np.unique(y, return_counts = True))

Random_state = 8888
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.2, 
                                                    random_state=Random_state, 
                                                    stratify=y)


#2. 모델구성
import xgboost as xgb
es = xgb.callback.EarlyStopping(
    rounds = 50,
    data_name = 'validation_0'
)
model = xgb.XGBClassifier(
    n_estimators=1000,
    random_state=Random_state,
    callbacks = [es])
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)])
print("score", model.score(x_test, y_test))

#3. SelectionFromModel
thresholds = np.sort(model.feature_importances_)
from sklearn.feature_selection import SelectFromModel
for i in thresholds:
    select = SelectFromModel(
        model, threshold = i, prefit = False
    )

    # x에 적용
    select_x_train = select.transform(x_train)
    select_x_test = select.transform(x_test)

    select_model = XGBClassifier(
                    n_estimators = 1000,
                    max_depth= 6, 
                    gamma = 0,
                    min_child_weigt = 0,
                    subsample = 0.4, 
                    # eval_metric = 'logloss', #이진분류: logloss, 다중분류:mlogloss
                    reg_alpha = 0, #L1규제, 절대값
                    reg_lamda = 1, #L2규제, 제곱합
                    # callbacks = [es],
                    # random_state = 412
    )
    from sklearn.metrics import accuracy_score
    select_model.fit(select_x_train, y_train)
    select_y_pre = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pre)

    print('Trech', i, 'n', select_x_train.shape[1], 'acc', score)
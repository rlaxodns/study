import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=423, 
                                                    stratify=y
                                                    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
import xgboost as xgb
es = xgb.callback.EarlyStopping(
    rounds = 50, 
    # metric_name = 'logloss',
    data_name = 'validation_0',
    save_best = True
)

model = XGBClassifier(
    n_estimators = 1000,
    max_depth= 6, 
    gamma = 0,
    min_child_weigt = 0,
    subsample = 0.4, 
    # eval_metric = 'logloss', #이진분류: logloss, 다중분류:mlogloss
    reg_alpha = 0, #L1규제, 절대값
    reg_lamda = 1, #L2규제, 제곱합
    callbacks = [es],
    random_state = 412
)

#3. 훈련
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)], 
          verbose = 1)

#4. 평가 및 예측
result = model.score(x_test, y_test)
print("최종점수", result)

y_pre = model.predict(x_test)
acc = accuracy_score(y_test, y_pre)
print("정확도", acc)

# 피쳐의 기여도 출력
# print(model.feature_importances_)
"""
[0.06601114 0.01886232 0.00607865 0.04149321 0.01342379 0.01660465
 0.01594858 0.04769487 0.01533063 0.01469479 0.00559003 0.00811798
 0.01773253 0.01838299 0.00766198 0.01361636 0.00854716 0.00617906
 0.01049663 0.01026521 0.01849999 0.02100686 0.10245489 0.27257675
 0.01966736 0.00392299 0.03089919 0.14804676 0.01526721 0.0049254 ]"""

thresholds = np.sort(model.feature_importances_)
# print(thresholds)

from sklearn.feature_selection import SelectFromModel
for i in thresholds:
    selection = SelectFromModel(
        model, threshold = i, prefit = False
    )
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

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
    select_model.fit(select_x_train, y_train, 
                    #  eval_set=[(select_x_test, y_test)]
                     )
    
    select_y_pre = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pre)

    print('Trech = %.3f, n=%d, acc:%.2f%%' %(i, select_x_train.shape[1], score*100))
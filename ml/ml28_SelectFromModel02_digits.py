from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_digits()
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
model = RandomForestClassifier(random_state=Random_state)
model.fit(x_train, y_train)


#3. selectFromModel
from sklearn.feature_selection import SelectFromModel
thresholds = np.sort(model.feature_importances_)

for i in thresholds:
    selet = SelectFromModel(
        model, threshold = i, prefit = False
    )
    
    sel_x_train = selet.transform(x_train)
    sel_x_test = selet.transform(x_test)

    sel_model = XGBClassifier(
        n_estimators = 100,
        Random_state = 4511,
    )

    from sklearn.metrics import accuracy_score
    sel_model.fit(sel_x_train, y_train)
    sel_y_pre = sel_model.predict(sel_x_test)
    acc = accuracy_score(y_test, sel_y_pre)

    print('thre', i, 'n', sel_x_train.shape[1], 'acc', acc)
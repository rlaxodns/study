from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


#1. 데이터
x, y = load_iris(return_X_y=True)
# print(x.shape, y.shape) #(150, 4) (150,)


Random_state = 8888
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.2, 
                                                    random_state=Random_state, 
                                                    stratify=y)


#2. 모델구성
model1 = DecisionTreeClassifier(random_state=Random_state)
model2 = RandomForestClassifier(random_state=Random_state)
model3 = GradientBoostingClassifier(random_state=Random_state)
model4 = XGBClassifier(random_state = Random_state)

models = [model1, model2, model3, model4]


print(Random_state)
for i in models:
    i.fit(x_train, y_train)
    print('========================', i.__class__.__name__, 'Random_state = ',Random_state, "======================")
    print('acc', i.score(x_test, y_test))
    print(i.feature_importances_)

"""
random_state = 777
======================== DecisionTreeClassifier(random_state=777) ======================
acc 0.9333333333333333
[0.         0.         0.06673611 0.93326389]
======================== RandomForestClassifier(random_state=777) ======================
acc 0.9333333333333333
[0.08215999 0.02226447 0.44785995 0.44771559]
======================== GradientBoostingClassifier(random_state=777) ======================
acc 0.9333333333333333
[0.00570895 0.00441326 0.51558481 0.47429298]
======================== XGBClassifier ======================
acc 0.9
[0.01801433 0.02953498 0.6262756  0.3261751 ]
"""


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import sklearn
from sklearn.model_selection import train_test_split

print(sklearn.__version__)
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
#1. 데이터
Random_state = 8888

d1 = load_breast_cancer()
d2 = load_digits()
d3 = load_wine()

datasets = [d1, d2, d3]
for dataset in datasets:
    x = dataset.data
    y = dataset.target
    
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

    for model in models:
        model.fit(x_train, y_train)
        print('========================', model.__class__.__name__,'Random_state = ',Random_state, "======================")
        print('acc', model.score(x_test, y_test))
        print(model.feature_importances_)
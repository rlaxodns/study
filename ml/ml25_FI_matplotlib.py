from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target
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


import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(i):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), i.feature_importances_,
            align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    plt.title(i.__class__.__name__)

plot_feature_importances_dataset(i)
plt.show()
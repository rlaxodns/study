from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd

#1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(150, 4) (150,)

df = pd.DataFrame(x, columns=dataset.feature_names)
df['Target'] = y

print("=======================상관계수 히트맵=======================")
print(df.corr())
"""
                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
Target                      0.782561         -0.426658           0.949035          0.956547  1.000000
"""
# 상관계수와 다중공선성에 대한 유의

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

sns.heatmap(data=df.corr(),
            square = True,
            annot= True,
            cbar = True)

plt.show()
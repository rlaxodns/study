from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

df['target'] = datasets.target

# df.boxplot()
# df.plot.box()
# plt.show()  # population의 데이터에 이상치 다수 존재

print(df.info())
print(df.describe())

import matplotlib.pyplot as plt
# df['Population'].boxplot() 시리즈에서는 이건 안된다.
# df['Population'].plot.box()

# df['Population'].hist(bins=50)
# plt.show()

x = df.drop(['target'], axis=1).copy()
y = df['target']


###########x의 population만 로그변환#######################
# 
x['Population'] = np.log1p(x['Population'])  # 지수변환 np.exp1m
###########################################################
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7, 
    random_state=1234,
)

################y로그변환########################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
# # ################################################

#2. 모델
model = LinearRegression()
    # random_state=1234,
                        #   max_depth=5,
                        #       min_samples_split=3,)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
from sklearn.metrics import r2_score
score = model.score(x_test, y_test)
print('score :', score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2', r2)

"""
<로그변환전>
score : 0.6444776255138276
r2 0.6444776255138276
<y후>
score : 0.6516449105475506
r2 0.6516449105475506
<x후>
score : 0.6444604594908349
r2 0.6444604594908349
<>
score : 0.6516440640883645
r2 0.6516440640883645


<리니어>
score : 0.6018146224164895
r2 0.6018146224164895

<>
score : 0.625128342532838
r2 0.625128342532838
<>
score : 0.6019282050160234
r2 0.6019282050160234
<>
score : 0.6248804583725058
r2 0.6248804583725058
"""
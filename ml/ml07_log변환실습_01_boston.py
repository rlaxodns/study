# 보스턴의 데이터를 다운받아서 보스턴의 집값 예측

from tensorflow.keras.models import Sequential
from keras.layers import Dense
import sklearn as sk
import numpy as np
print(sk.__version__)  #0.24.2

#1. 데이터
from sklearn.datasets import load_boston
dataset = load_boston() 
# print(dataset.shape)
print(dataset.DESCR)  # sklearn에서 .describe()와 동일한 데이터의 평균 등을 설명하는 함수
print(dataset.feature_names)  #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
                              #   'B' 'LSTAT']

import matplotlib.pyplot as plt
# dataset.boxplot() 시리즈에서는 이건 안된다.
# dataset.plot.box()

x = dataset.data
y = dataset.target  #x와 y를 데이터 상에서 분리


print(x.shape) #(506, 13)
print(y.shape) #(506,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size = 0.3, shuffle=True, random_state=6265)       

##################################################
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
##################################################

#2. 모델
model = Sequential()
model.add(Dense(100, input_dim = 13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_split=0.2, verbose=1)

#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict([x])

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print(loss)
print(r2)

"""
1
random=256
epo=1000
26.968708038330078
0.72588187868526

2
ran =625
epo=1000
17.805763244628906
0.7536866961834869

3.
26.192617416381836
0.6945241458236902
"""
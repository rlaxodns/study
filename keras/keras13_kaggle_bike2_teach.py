# x의 공백을 예측하여, 해당 예측치를 가지고 y의 값을 예측

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터 준비

train_csv = pd.read_csv('C:\\ai5\\_data\\bike-sharing-demand\\train.csv', index_col=0)
test_csv = pd.read_csv('C:\\ai5\\_data\\bike-sharing-demand\\test.csv', index_col=0)
sample_csv = pd.read_csv('C:\\ai5\_data\\bike-sharing-demand\\sampleSubmission.csv', index_col=0)

######결측치 확인#############
print(train_csv.isna().sum())
print(test_csv.isna().sum())
print(sample_csv.isna().sum())

##1. x와 y의 분리
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv[['casual', 'registered']]

##2. 훈련, 검증 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
print(x_train.shape, y_train.shape)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_dim = 8))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='linear'))

#3. 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10)

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([test_csv])

y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

print('결과', result)
print('오차', loss)
print('r2', r2)
print(result.shape) #(6493, 2)

##. 결과를 test_csv에 입력
# 판다스의 데이터라고 하더라도 예측치는 넘파이 형태로 출력
print("test_csv타입:", type(test_csv)) #test_csv타입: <class 'pandas.core.frame.DataFrame'>
print("result의 타입:", type(result)) #result의 타입:  <class 'numpy.ndarray'> # 넘파이에는 순수한 데이터만 존재
                                    #넘파이에는 인덱스와 컬럼명이 존재하지 않는다. 따라서 컬럼명을 명시하면서 데이터 입력

test2_csv = test_csv
print(test2_csv.shape) #(6493, 8)

test2_csv[['casual', 'registered']] = result
print(test2_csv) #[6493 rows x 10 columns]
test2_csv.to_csv('C:\\ai5\\_data\\bike-sharing-demand\\test2.csv')
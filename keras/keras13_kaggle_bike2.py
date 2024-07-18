# 기존의 캐글 데이터에서 
# 1. train_csv의 y를 casual과 registered로 잡는다.add()
# 그래서 훈련을 해서 test_csv casual과 registered를 predict를 한다.

# 2. test_csv에 캐주얼과 레지스터 컬럼을 합쳐서 count를 예측한다

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터 불러오기
train =  pd.read_csv('C:\\ai5\\_data\\bike-sharing-demand\\train.csv', index_col=0)
test = pd.read_csv('C:\\ai5\\_data\\bike-sharing-demand\\test.csv', index_col=0)
submission = pd.read_csv('C:\\ai5\\_data\\bike-sharing-demand\\sampleSubmission.csv', index_col=0)


# train 데이터 프레임 정리
x = train.drop(['casual', 'registered','count'], axis = 1)
y = train[['casual', 'registered']]
print(y.columns)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=10)
print(x_train.shape, y_train.shape)
print(y_train)


#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = 8))
model.add(Dense(2, activation='linear'))


#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=10, batch_size=16)

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(test)

y_predict = model.predict([x_test])
test['casual'] = 0
test['registered'] = 0
print(test.columns)
r2 = r2_score(y_test, y_predict)
print(result)

test[['casual','registered']] = result
print(test)

test.to_csv('C:\\ai5\\_data\\bike-sharing-demand\\test0001.csv')
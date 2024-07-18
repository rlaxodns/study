import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터 불러오기
train =  pd.read_csv('C:\\ai5\\_data\\bike-sharing-demand\\train.csv', index_col=0)
test0001 = pd.read_csv('C:\\ai5\\_data\\bike-sharing-demand\\test0001.csv', index_col=0)
submission = pd.read_csv('C:\\ai5\\_data\\bike-sharing-demand\\sampleSubmission.csv', index_col=0)

# train 데이터 프레임 정리
x = train.drop(['count'], axis = 1)
y = train[['count']]
print(y.columns)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=10)
print(x_train.shape, y_train.shape)
print(y_train)


#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim = 10))
model.add(Dense(1, activation='linear'))


#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=10, batch_size=16)

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
test_count = model.predict(test0001)

y_predict = model.predict([x_test])

print(test0001.columns)
r2 = r2_score(y_test, y_predict)
print(test_count)

submission[['count']] = test_count
print(submission)

submission.to_csv('C:\\ai5\\_data\\bike-sharing-demand\\submission0001_1.csv')
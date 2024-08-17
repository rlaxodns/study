from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, Conv1D

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 1000,  # 단어사전의 숫자
                                                        #  maxlen = 100,
                                                         test_split = 0.2,
                                                        )
# print(x_train)
# print(x_train.shape, x_test.shape) #(8982,) (2246,)
# print(y_train.shape, y_test.shape) #(8982,) (2246,)
# print(y_train)

# print(np.unique(y_train))
# # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
# #  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# print(len(np.unique(y_train))) #46
 
# print(type(x_train)) #<class 'numpy.ndarray'>
# print(type(x_train[0])) #<class 'list'> ---> 리스트 데이터를 넘파이로 변환해야함
# print(len(x_train[0]), len(x_train[1])) # 87 56.... pad_sequences필요

print("뉴스기사의 최대길이 :", max(len(i) for i in x_train)) #뉴스기사의 최대길이 : 2376
print("뉴스기사의 최소길이 :", min(len(i) for i in x_train)) #뉴스기사의 최소길이 : 13
print("뉴스기사의 평균길이 :", sum(map(len,x_train))/len(x_train)) #뉴스기사의 평균길이 : 145.5398574927633

from keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, maxlen=100, truncating='pre')

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 스케일링
from sklearn.preprocessing import StandardScaler, MinMaxScaler
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Embedding(1000, 512))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(46, activation='softmax'))

#3. 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train,
          epochs=10, 
          batch_size=128)

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)

print(loss)
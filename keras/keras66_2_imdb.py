from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, Conv1D


(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = 1000,  # 단어사전의 숫자
    # maxlen = 100,
    # test_split = 0.2,
    )

print(x_train.shape, x_test.shape)
print(np.unique(y_train)) #[0 1]
# print( max(len(i) for i in x_train)) #2494
# print( min(len(i) for i in x_train)) #11
# print( sum(map(len,x_train))/len(x_train)) #238.71364

from keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=100, padding='pre', truncating='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre', truncating='pre')
print(x_train.shape, x_test.shape) #(25000, 2494) (25000, 2315)

#스케일링
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# std = StandardScaler()
# x_train = std.fit_transform(x_train)
# x_test = std.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Embedding(1000, 128))
model.add(Conv1D(64, 2))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일 및 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train,
          epochs=10, 
          batch_size=128)

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)

print(loss)
#[0.5360210537910461, 0.8173999786376953]
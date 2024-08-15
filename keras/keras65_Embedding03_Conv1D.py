from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요', 
    '재미없어요', '너무 재미없다', '참 재밋네요', 
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])
y = ["태운이 참 재미없다."]

token = Tokenizer()
token.fit_on_texts(docs)
token.fit_on_texts(y)

x = token.texts_to_sequences(docs)
y = token.texts_to_sequences(y)

#### pad_sequence ###
from keras.preprocessing.sequence import pad_sequences
x = pad_sequences(x)
y = pad_sequences(y, 5)

# print(x, x.shape) #(15, 5)
# print(y, y.shape) #(1, 5)

###3차원 데이터로 reshape
x = x.reshape(15, 5, 1)
y = y.reshape(1, 5, 1)

# 모델 구성
model = Sequential()
model.add(Conv1D(16,2, input_shape=(5,1)))
model.add(Conv1D(16,2))
model.add(Flatten())
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

# 컴파일 및 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam')
model.fit(x, labels,
          epochs=100, 
          batch_size=1)

# 평가 및 예측
loss = model.evaluate(x, labels)
result = np.round(model.predict(y))

print(loss, result)
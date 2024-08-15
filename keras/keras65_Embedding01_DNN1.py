# # 길이가 다른 문제를 해결하는 방안은 가장 긴 문장의 길이를 맞추기 위해서 짧은 문장에 0을 채워 넣는다.
# # 문장의 의미가 주로 뒤에 많이 가지고 있기 때문에 0을 앞쪽에 채워 넣는 것이 영향력을 최소화할 수 있다.
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# paded = pad_sequences(x,
#                     #   padding='post',    #기본값이 앞으로 0을 채워 넣음, post는 뒤로 0넣음, 'pre'는 앞에 0넣음
#                     #   maxlen=5,          #길이가 5보다 짧은 문서들은 0으로 패딩되고, 기존에 5보다 길었다면 데이터가 손실된다.
#                     #   truncating='post', #뒤의 단어가 삭제되도록 하고싶다면 truncating이라는 인자를 사용합니다. truncating='post'를 사용할 경우 뒤의 단어가 삭제
#                     #   value= 다른 숫자   #pad_sequences의 인자로 value를 사용하면 0이 아닌 다른 숫자로 패딩이 가능
#                     ) 

from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

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

from tensorflow.keras.preprocessing.sequence import pad_sequences
x = pad_sequences(x)
y = pad_sequences(y, 5)
# print(x, x.shape) #(15, 5)
print(y, y.shape) #(1, 5)

# 모델
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(16, input_shape=(5,)))
model.add(Dense(1, activation='sigmoid'))

# 컴파일 및 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x, labels,
          epochs=10,
          batch_size=1)

# 평가 및 예측
loss = model.evaluate(x, labels)
result = model.predict(y)

print(loss)
print(result)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten
import numpy as np

#1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화예요',
    '추천하고 싶은 영화입니다', '한 번 더 보고 싶어요', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요', 
    '재미없어요', '너무 재미없다', '참 재밋네요', 
    '준영이 바보', '반장 잘생겼다', '태운이 또 구라친다',
    '동해물과 백두산이 마르고 닳도록 하나님이 보우하사',
    '우리나라 만세 무궁화 삼천리 화려강산 대한사람 대한으로',
    '영풍문고는 책이 별로 없다', '집에 너무 너무 가고 싶다',
    '반장 주말에 출근을 사기친다.'
]

labels = np.array([1,1,1,1,1,0,0,0,0,0,
                   0,1,0,1,0,1,1,0,0,1])
x_pre = '참 잘생겼다 무궁화'

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
"""
{'너무': 1, '참': 2, '반장': 3, '재미있다': 4,
'최고에요': 5, '잘만든': 6, '영화예요': 7, '추천하고': 8,
'싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13,
'보고': 14, '싶어요': 15, '글쎄': 16, '별로에요': 17, '생각보다': 18,
'지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22,
'재미없다': 23, '재밋네요': 24, '준영이': 25, '바보': 26, '잘생겼다': 27,
'태운이': 28, '또': 29, '구라친다': 30, '동해 물과': 31, '백두산이': 32, '마르고': 33, 
'닳도록': 34, '하나님이': 35, '보우하사': 36, '우리나라': 37, '만세': 38, '무궁화': 39, '삼천리': 40, '화려강산': 41, '대한사람': 42, '대한으로': 43, '영풍문고는': 44, 
'책이': 45, '별로': 46, '없다': 47, '집에': 48, '가고': 49, '싶다': 50, '주말에': 51, '출근을': 52, '사기친다': 53}"""

x = token.texts_to_sequences(docs)
y = token.texts_to_sequences([x_pre])
# print(x.shape)

x = pad_sequences(x,
                  maxlen=5,
                  truncating='pre')
y = pad_sequences(y, maxlen=5)
# print(x.shape, y.shape) #(20, 5) (13, 5)
# print(y)

# 
# 원핫인코딩
x = to_categorical(x)
y = to_categorical(y, num_classes=54)
print(x.shape, y.shape) #(20, 5, 54) (13, 5, 54)


# 모델 구성
model = Sequential()
model.add(Conv1D(16,2, input_shape=(5,54), activation='relu'))
model.add(Conv1D(16,2, activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 컴파일 및 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam')
model.fit(x, labels,
          epochs=100, 
          batch_size=1)

# 평가 및 예측
loss = model.evaluate(x, labels)
result = model.predict(y)

print(loss, result)

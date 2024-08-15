# 앙상블을 하려면 행의 갯수를 맞춰야한다.
## 오늘의 과제 class & instance에 대해서 정의하고 제출, ex) 인스턴스를 생성하였다.

# Tokenizer
# 음절, 어절, 형태소 == 단어기반을 통해서 수치화를 통해 분석에 용이하게 만들어줌

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

text = "나는 지금 진짜 진짜 매우 매우 맛있는 김밥을 엄첨 마구 마구 마구 마구 먹었다."
text1 = "Let's go to the party"
text2 = "동해물과 백두산이 마르고 닳도록 하나님이 보우하사 우리나라 만세"
text3 = "바우 와우 와우"

token = Tokenizer()
token.fit_on_texts([text])
# print(token.word_index) 
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '지금': 5, '맛있는': 6, '김밥을': 7, '엄첨': 8, '먹었다': 9}


token.fit_on_texts([text1])
# print(token1.word_index)
# {"let's": 1, 'go': 2, 'to': 3, 'the': 4, 'party': 5}

token.fit_on_texts([text2])
# print(token2.word_index)
# {'동해물과': 1, '백두산이': 2, '마르고': 3, '닳도록': 4, '하나님이': 5, '보우하사': 6, '우리나라': 7, '만세': 8}

token.fit_on_texts([text3])
# print(token3.word_index)
# {'와우': 1, '바우': 2}

# print(token.word_counts)
# OrderedDict([('나는', 1), ('지금', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('김밥을', 1), ('엄첨', 1), ('마구', 4), ('먹었다', 1)])

## 문장을 수치로 변환하여 분석에 용이하게 만들어주는 함수
x = token.texts_to_sequences([text])
print(x)

# [[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
# print(x.shape) # 리스트는 shape이 없음
# 현재의 수치화된 데이터를 원핫인코딩을 통해 주소를 부여해야 분석에 용이하게 된다

# # ## sklearn의 OneHotEncoder
# x = np.array(x).reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# x = ohe.fit_transform(x)
# print(x)
# print(x.shape)
"""[[4, 5, 2, 2, 3, 3, 6, 7, 8, 1, 1, 1, 1, 9]]
[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1.]]
(14, 9)
"""


### keras의 to_categorical
from tensorflow.keras.utils import to_categorical
x = to_categorical(x) # num_classes = 10으로 숫자 갯수만큼으로 맞춰줌
x = x[:,:,1:].reshape(14,9)
print(x.shape)
print(x)
print(type(x))
"""
(1, 14, 10)
[[[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]]"""

### pandas의 get_dummies
# x = [item for sublist in x for item in sublist] #x = np.array(x).reshape(-1,)로도 가능
# x = pd.Series(x)
# x = pd.get_dummies(x)
# print(x.shape)
# print(x)
# print(type(x))
"""
(14, 9)
    1  2  3  4  5  6  7  8  9
0   0  0  0  1  0  0  0  0  0
1   0  0  0  0  1  0  0  0  0
2   0  1  0  0  0  0  0  0  0
3   0  1  0  0  0  0  0  0  0
4   0  0  1  0  0  0  0  0  0
5   0  0  1  0  0  0  0  0  0
6   0  0  0  0  0  1  0  0  0
7   0  0  0  0  0  0  1  0  0
8   0  0  0  0  0  0  0  1  0
9   1  0  0  0  0  0  0  0  0
10  1  0  0  0  0  0  0  0  0
11  1  0  0  0  0  0  0  0  0
12  1  0  0  0  0  0  0  0  0
13  0  0  0  0  0  0  0  0  1"""

import numpy as np

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
               [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
                [9,8,7,6,5,4,3,2,1,0]])

y = np.array([1,2,3,4,5,6,7,8,9,10])


#2. 모델구성
# 지금까지의 모델은 순차적인 형태의 구성
# 모델을 (1)순차형으로 만들 것인가 (2) 함수형으로 만들것인가의 새로운 문제
# 성능은 동일하지만, 표현방식이 다를 뿐이다
from keras.models import Model, Sequential
from keras.layers import Input, Dense

#2-1. 모델구성(순차형)
# model = Sequential()
# model.add(Dense(10, input_shape = (3,)))
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(1))

# Model: "sequential"
"""________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 10)                40

 dense_1 (Dense)             (None, 9)                 99

 dense_2 (Dense)             (None, 8)                 80

 dense_3 (Dense)             (None, 7)                 63

 dense_4 (Dense)             (None, 1)                 8

=================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0
"""

# #2-2. 모델구성
input1 = Input(shape=(3,))
dense1 = Dense(10, name="king")(input1)
dense2 = Dense(9, name="kingtaewoon")(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(7)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs = input1, outputs = output1)
"""
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 3)]               0

 dense (Dense)               (None, 10)                40

 dense_1 (Dense)             (None, 9)                 99        

 dense_2 (Dense)             (None, 8)                 80

 dense_3 (Dense)             (None, 7)                 63

 dense_4 (Dense)             (None, 1)                 8

=================================================================
Total params: 290
Trainable params: 290
Non-trainable params: 0
_________________________________________________________________
"""
model.summary()
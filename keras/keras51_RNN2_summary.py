# RNN은 시계열 데이터에 대해서 분석한다
# 통상적으로 종속변수(Y)의 값을 주어지지 않기 때문에 회귀분석을 할 수 없어서 '지도학습'을 할 수 없다.
# 그렇기 때문에 독립변수(x)를 '비지도학습'을 적용하여, 예측을 한다. 
# ex) 연속형 독립변수들 내에서 일정범위(타임스텝)를 통해 종속변수를 만들어 활용

# 따라서, 타임스텝을 나누는 과정이 모델의 예측과 정확성에 영향을 미친다. 
# RNN (순환신경망모델) ==up grade==> LSTM, GRU  ==> transformer

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3], #각각의 행은 timesteps
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],]
             )

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) #(7, 3) (7,) 2차원 형태의 데이터기 때문에 rnn구동 위한 차원 증가의 필요

x = x.reshape(
    x.shape[0],
    x.shape[1], 
    1
) # RNN 사용을 위한 차원 증가
print(x.shape) #(7, 3, 1) #3D tensor with shape (batch_size, timesteps, features)

#2. 모델 구성
model = Sequential()

model.add(SimpleRNN(20, input_shape = (3,1), activation='relu')) # 행무시, 열우선 input_shape=(timesteps, features)
# 연산량 계산법 = (units*units) + (units*dim) +bias =units *(units + features + bias)
model.add(Dense(10, activation='relu'))                     
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (None, 20)                440

 dense (Dense)               (None, 10)                210

 dense_1 (Dense)             (None, 10)                110

 dense_2 (Dense)             (None, 10)                110

 dense_3 (Dense)             (None, 7)                 77

 dense_4 (Dense)             (None, 1)                 8

=================================================================
Total params: 955
Trainable params: 955
Non-trainable params: 0
_________________________________________________________________
"""
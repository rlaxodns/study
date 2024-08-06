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
# model.add(SimpleRNN(20, input_shape = (3,1), activation='relu')) # 행무시, 열우선 input_shape=(timesteps, features)
model.add(SimpleRNN(10, input_length = 3, # 타임스텝의 길이
                     input_dim = 1))      # 피쳐의 크기                 
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, 
          epochs = 1024, batch_size=2, validation_split=0.1)

#4. 평가 및 예측
loss = model.evaluate(x, y)
x_pre = np.array([8,9,10]).reshape(1,3,1)

result = model.predict(x_pre) # [[[8],[9],[10]]] 

print("loss :", loss, "result :", result)

# loss : 4.708781943918439e-06 result : [[11.008226]]
# loss : 0.00010336581181036308 result : [[10.9346]]
# <LSTM> loss : 0.017493069171905518 result : [[10.419899]]
# loss : 3.5975390346720815e-05 result : [[10.982119]]
# <GRU> loss : 0.0001437823084415868 result : [[10.936319]]
# loss : 4.255393287166953e-05 result : [[10.973548]]
"""
keras.layers.SimpleRNN(
    units,
    activation="tanh",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    seed=None,
    **kwargs
    """
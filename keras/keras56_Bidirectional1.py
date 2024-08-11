"""
LSTM에 방향성이 존재, 
ex) 날씨를 체크하기 위해서 날짜의 순서로 확인할 수도 있으나, 역순으로도 확인할 수 있다.
훈련을 시킬때, 방향성을 가지고 순서로 훈련을 진행, 역순을 통해서도 훈련이 가능
이로 인해 통상적으로 1회의 훈련을 진행하던 것을 순차 및 역순 훈련을 통해서 2회 진행가능하다 (= 양방향RNN)
"""
#51_1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional

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
model.add(Bidirectional(GRU(20), input_shape = (3,1)))   #TypeError: __init__() missing 1 required positional argument: 'layer'
# 대문자는 통상 클래스, 소문자는 통상 함수 // Bidirectioal = wrapping class(RNN을 wrapping하는 클래스)
# 기존의 RNN Layer에 연산량이 2배가 된다 
# 행무시, 열우선 input_shape=(timesteps, features)
model.add(Dense(10, activation='relu'))                     
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()

"""
 simple_rnn (SimpleRNN)      (None, 20)                440
 bidirectional (Bidirectiona  (None, 40)               880
 
 lstm (LSTM)                 (None, 20)                1760
 bidirectional (Bidirectiona  (None, 40)               3520

  gru (GRU)                   (None, 20)                1380
  bidirectional (Bidirectiona  (None, 40)               2760
"""

# #3. 컴파일 및 훈련
# model.compile(loss = 'mse', optimizer='adam')
# model.fit(x, y, 
#           epochs = 1024, batch_size=2, validation_split=0.1)

# #4. 평가 및 예측
# loss = model.evaluate(x, y)
# x_pre = np.array([8,9,10]).reshape(1,3,1)

# result = model.predict(x_pre) # [[[8],[9],[10]]] 

# print("loss :", loss, "result :", result)
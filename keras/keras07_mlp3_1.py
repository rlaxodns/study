import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([range(10), range(21,31), range(201, 211)]) #x1,x2,x3의 데이터 존재
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]])  #y1, y2의 데이터 존재

print(x.shape) #(3, 10)
print(y.shape) #(2,10)

x=x.T
y=np.transpose(y)

print(x.shape) #(10,3)
print(y.shape) #(10,2)

## [실습]
# [10,31,211]

# 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(2))  #y1, y2 값으로 출력

# 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 예측 및 출력
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])

print("로스 \n", loss)
print("예측 \n", result)

"""
 로스 
  3.91949462890625
 예측
  [[7.9246426 // 4.532321 ]]
"""

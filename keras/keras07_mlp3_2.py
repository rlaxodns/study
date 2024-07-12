import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. data
x = np.array([range(10), range(21,31), range(201, 211)]) #x1,x2,x3의 데이터 존재

y = np.array ([[1,2,3,4,5,6,7,8,9,10],
                [10,9,8,7,6,5,4,3,2,1],
                [9,8,7,6,5,4,3,2,1,0]]) #y1, y2, y3의 데이터가 존재

print(x.shape)
print(y.shape)

x=x.T
y=y.T

print(x.shape)
print(y.shape)

#2. model
model= Sequential()
model.add(Dense(10, input_dim=3)) #인풋으로 3개의 데이터 묶음을 입력
model.add(Dense(3))  #아웃풋으로 3개의 데이터 출력

#3 compile
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

#4 predict
loss = model.evaluate(x, y)
result = model.predict([[10,31,211]]) # x1=10, x2=31, x3=211을 입력하여
# y1,y2,y3의 값을 출력

print("loss\n", loss)
print("result\n", result)

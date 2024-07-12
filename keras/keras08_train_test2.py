import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense

# data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#[실습] 넘파이 리스트의 슬라이싱, 7:3으로 잘라라

x_train = x[:7] #x_train = x[0:7] or x_train = x[:-3]
y_train = y[:7] 

x_test = x[7:]
y_test = y[7:]  #첫 인덱스와 마지막 인덱스는 생략 가능


# model
model = Sequential()
model.add(Dense(1, input_dim = 1))

# compile
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#predict
loss = model.evaluate(x_test, y_test)
result = model.predict([11])

print(loss)
print(result)
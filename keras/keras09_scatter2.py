import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense

# data
x = np.array([1,2,3,4,6,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=270)

# 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 예측 및 평가
loss = model.evaluate(x_test, y_test)
result = model.predict([x])

print(loss)
print(result)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, result, color = 'red')
plt.show()

"""
오차값을 가지고 좋고 나쁨을 판단할 때에 있어 
무엇이 더 좋은지 판단하기 어려운 경우가 있어서 신뢰하기 어려워서
보조지표로) accuracy(분류)와 R2(선형회귀)"""
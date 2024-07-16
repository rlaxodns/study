"""
시각화
"""
#사이킷런

import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# data
x = np.array([1,2,3,4,6,5,7,8,9,10])
y = np.array([1,2,3,7,5,6,7,8,6,10])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                    test_size=0.3,  # 테스트 사이즈의 기본값 = 0.25
                                    train_size=0.7,  # 트레인 사이즈의 기본값 = 0.75
                                    # 테스트+트레인=1 이상을 넘어가는 것은 안된다
                                    # 하지만 1 미만인 경우에는 데이터를 소실하면서 출력
                                    shuffle=True,  #디폴트 값이 참
                                    random_state=1004
                                    )
# train_size =  검증 데이터의 사이즈 지정 / shupple = True(데이터를 랜덤하게 섞음)
# random_state = 랜덤 값을 고정하는 것 / 난수의 사전에서의 섞는 방법을 지정하는 것

print('x_train', x_train)
print('x_test', x_test)
print('y_train', y_train)
print('y_test', y_test)

print(x_train.shape) #(7,)
print(x_test.shape)  #(3,)

# model
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# compile
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#predict
loss = model.evaluate(x_test, y_test)
result = model.predict([x])

print(loss)
print(result)

import matplotlib.pyplot as plt
plt.scatter(x, y)  # 원 데이터의 위치
plt.plot(x, result, color ='red') # x에 대한 결과값
plt.show()  # 시각화
# 15-4 copy
# 훈련에 대한 시간을 확인 하는 이유, 
# 큰 데이터의 훈련을 돌리면서의 시간의 효율성 문제를 해결할 수 있음
# 다만 첫 번째에 대한 시간은 확인할 수 없음 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import time

# 데이터 구성
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                            random_state=10)
print(x_train)

# 모델 
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))


# 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2)
end_time = time.time()


# 평가 및 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([18])

print(loss)
print(result)
print(round(end_time - start_time,2), "초")  #round(x, 2) 소수점 둘째자리까지 반올림하여라

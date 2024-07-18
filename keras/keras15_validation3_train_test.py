import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터 구성
x = np.array(range(0, 17))
y = np.array(range(0, 17))

## train_test_split으로 3분할 하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.25,
                                                random_state=10)
print(x_train)
print(x_test)

# 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

# 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, 
          validation_data=(x_vali, y_vali))

#평가 및 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([17])

print(loss)
print(result)
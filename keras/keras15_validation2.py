import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터 구성
x = np.array(range(0, 17))
y = np.array(range(0, 17))
print(x)

x_train = x[:12]
print(x_train)
y_train = y[:12]

x_vali = x[12:15]
print(x_vali)
y_vali = y[12:15]

x_test = x[15:17]
print(x_test)
y_test = y[15:17]

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_vali,y_vali))

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([17])

print(loss)
print(result)
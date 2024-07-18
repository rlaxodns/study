import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 데이터 구성
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, 
                                            random_state=10)
print(x_train)

# 모델 
model = Sequential()
model.add(Dense(1, input_dim=1))

# 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1,
          validation_split=0.2)

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([18])

print(loss)
print(result)
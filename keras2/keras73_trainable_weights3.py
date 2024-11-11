import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)

#1. 데이터
# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])

x = np.array([1])
y = np.array([1])
print(x.shape)

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

################################
# model.trainable = False # 동결**
model.trainable = True
################################
print(model.weights)
print('=========================')

#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose = 0)

#4. 평가 및 예측
y_pred = model.predict(x)
print(y_pred)
print('=========================')
print(model.weights)
print('=========================')
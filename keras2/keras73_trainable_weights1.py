import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(3333)
np.random.seed(3333)
print(tf.__version__)

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
print(x.shape)

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
print(model.weights)
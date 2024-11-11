import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer, Flatten
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)

#1. 데이터
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

#2. 모델
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(# weights='imagenet',
              include_top=False, # True가 디폴트 #False적용시 flatten()하단의 Dense layer가 사라짐
              input_shape=(32, 32, 3))

vgg16.trainable = True

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, 
          epochs = 1, 
          batch_size=120,
          validation_split=0.2)

#4. 평가 및 예측
from sklearn.metrics import accuracy_score
loss = model.evaluate(x_test, y_test)
y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_pre)

print("loss :", loss,
      "acc :", acc)
# loss : 3273.69970703125 acc : 0.0151
# loss : 3273.69970703125 acc : 1.0
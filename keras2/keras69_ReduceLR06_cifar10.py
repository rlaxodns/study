import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
import time
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
import random as rn
rn.seed(6265)
tf.random.set_seed(6265)
np.random.seed(6265)

#1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()

#x데이터 스케일링
x_train = x_train/255.
x_test = x_test/255.

# y원핫인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# print(y_train.shape) #(50000, 10)

# #2. 모델 구성
model = Sequential()
model.add(Conv2D(500, (4,4), input_shape = (32,32,3)))
model.add(Conv2D(250, (3,3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(100, (2,2), activation='relu'))
model.add(Conv2D(50, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 및 훈련
from tensorflow.keras.optimizers import Adam
lr = [0.1]

es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min', 
    patience=50, 
    verbose=1,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode = 'auto', 
    verbose=1, 
    save_best_only=True, 
    filepath=".//_save//keras35//keras35__save_cnn6_mnist00.hdf5"
)

    
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['acc'])
model.fit(x_train, y_train,
        epochs = 100,
        batch_size=128,
        validation_split=0.2,
        callbacks=[es, mcp])

#4. 평가 및 예측
print("===========출력==================")
loss = model.evaluate(x_test, y_test, verbose=0)
print('lr:{0},로스:{1}'.format(lr, loss[0]))
print('lr:{0},r2:{1}'.format(lr, loss[1]))

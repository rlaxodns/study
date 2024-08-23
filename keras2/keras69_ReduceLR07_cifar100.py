import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
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
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train, return_counts = True))


# x 데이터 스케일링
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
x_train = mms.fit_transform(x_train)
x_test = mms.fit_transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# y 원핫인코딩
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))

#2. 모델
model = Sequential()
model.add(Conv2D(25, (5,5), input_shape = (32,32,3))) #((커널사이즈*커널사이즈)*채널+1)*필터
model.add(Conv2D(12, (4,4)))         # 상당의 필터는 하단의 채널이 된다, 커널 사이즈가 사실상 CNN의 가중치가 된다
model.add(BatchNormalization()) 
model.add(Conv2D(20, (3,3)))
model.add(Conv2D(10, (2,2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

# #3. 컴파일 및 훈련
from tensorflow.keras.optimizers import Adam
lr = [0.1]

es = EarlyStopping(
    monitor='val_loss', 
    mode = 'min',
    patience=20,
    verbose=1, 
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose=1, 
    save_best_only=True,
    filepath='.//_save//keras35//keras35__save_cnn7_cifar100_00.hdf5'
)
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=lr),
                metrics=['acc'])
model.fit(x_train, y_train,
            epochs = 1000,
            batch_size=300,
            validation_split=0.2,
            callbacks=[es, mcp]
)

#4. 평가 및 예측
print("===========출력==================")
loss = model.evaluate(x_test, y_test, verbose=0)
print('lr:{0},로스:{1}'.format(lr[i], loss[0]))
print('lr:{0},r2:{1}'.format(lr[i], loss[1]))
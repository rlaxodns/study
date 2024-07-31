import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import time
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
model.add(Conv2D(25, (2,3),
                 padding='same', input_shape = (32,32,3))) #((커널사이즈*커널사이즈)*채널+1)*필터
model.add(MaxPooling2D())
model.add(Conv2D(12, (2,2), activation='relu', strides=2))         # 상당의 필터는 하단의 채널이 된다, 커널 사이즈가 사실상 CNN의 가중치가 된다
model.add(BatchNormalization()) 
model.add(Conv2D(20, (2,2),padding='same', activation='relu', strides=2))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(200, activation='relu'))

model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))

model.summary()

#3. 컴파일 및 훈련
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
    filepath='.//_save//keras35//keras35__save_cnn7_cifar100_02.hdf5'
)

model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train,
          epochs = 1000,
          batch_size=512,
          validation_split=0.1,
          callbacks=[es, mcp]
)

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = model.predict(x_test)

# print(type(y_test))
y_pre = np.argmax(y_pre, axis = 1).reshape(-1,1)
y_test = np.argmax(y_test, axis = 1).reshape(-1,1)

acc = accuracy_score(y_test, y_pre)

print(loss, acc)
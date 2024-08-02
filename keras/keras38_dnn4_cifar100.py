import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터 구성
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

## 데이터 스케일링
x_train = x_train/255.
x_test = x_test/255.

## 데이터 인코딩
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

#2. 모델 구성
model = Sequential()
model.add(Dense(512, input_shape = (32*32*3, )))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,     
    save_best_only=True,   
    filepath="C:\\ai5\\_save\\keras38\\keras38_04_.hdf5", 
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=2000, batch_size=64,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#4. 평가, 예측  
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)
print("걸린 시간 :", round(end-start,2),'초')


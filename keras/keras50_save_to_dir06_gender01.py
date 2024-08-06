import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#1. 데이터
 

x_train = np.load("C:\\ai5\\_data\\_save_npy\\save_gender\\x_train.npy",)
y_train =np.load("C:\\ai5\\_data\\_save_npy\\save_gender\\y_train.npy",)
x_test = np.load("C:\\ai5\\_data\\_save_npy\\save_gender\\x_test.npy",)
y_test = np.load("C:\\ai5\\_data\\_save_npy\\save_gender\\y_test.npy",)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, BatchNormalization, Flatten
model = Sequential()
model.add(Conv2D(32, kernel_size=(2,2), input_shape = (100,100,3), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(16, (2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(254, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 및 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='acc',
    mode = 'max', 
    patience=20, 
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor='acc',
    mode = 'auto', 
    verbose=1, 
    save_best_only=True,
    filepath="C:\\ai5\\_save\\keras49\\keras49_6_men_women.hdf5"
)

model.fit(x_train, y_train,
          epochs=1000,
          batch_size=126,
          validation_split=0.2,
          callbacks=[es, mcp])

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict(x_test))
y_test = np.round(y_test)

acc = accuracy_score(y_test, y_pre)

print(loss[0], acc)
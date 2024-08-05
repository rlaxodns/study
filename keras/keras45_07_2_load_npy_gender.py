import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#1. 데이터
np_path = "C:\\ai5\\_data\\_save_npy\\"
x_train = np.load(np_path + "ke4507_gender_x_train.npy")
y_train = np.load(np_path + "ke4507_gender_y_train.npy")

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size= 0.2, random_state=4343
)

print(x_train.shape)

#2. 모델 구성
# 레이어 1
model = Sequential()
model.add(Conv2D(32, (2,2), activation="relu", input_shape=(100,100,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 레이어 2 
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 레이어3
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
          
# Fully Connected 
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1,activation="sigmoid"))

#3. 컴파일 및 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=20,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath = "C:\\ai5\\_save\\keras45\\keras4507_0002.hdf5"
)

model.fit(x_train, y_train, 
          epochs = 1000,
          batch_size=512,
          validation_split=0.2,
          callbacks=[es, mcp])

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pre)

print(loss , acc)
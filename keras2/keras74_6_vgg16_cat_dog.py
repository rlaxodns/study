# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


#1. 데이터

x_train = np.load("C:\\ai5\\_data\\_save_npy\\save_cat_dog\\x_train.npy",)
y_train = np.load("C:\\ai5\\_data\\_save_npy\\save_cat_dog\\y_train.npy",)
x_test = np.load("C:\\ai5\\_data\\_save_npy\\save_cat_dog\\x_test.npy",)
y_test = np.load("C:\\ai5\\_data\\_save_npy\\save_cat_dog\\y_test.npy",)

from tensorflow.keras.applications import VGG16

vgg16 = VGG16(# weights='imagenet',
              include_top=False, # True가 디폴트 #False적용시 flatten()하단의 Dense layer가 사라짐
              input_shape=(100, 100, 3))

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))

model.summary()


#3. 컴파일 및 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam')
model.fit(x_train, y_train, 
          epochs = 1, 
          batch_size=120,
          validation_split=0.2)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=16)
print('loss :', loss)
# print('acc :', round(loss[1],5))

y_pre = model.predict(x_test, batch_size=16)


y_pre = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)
"""
false loss : 0.5054184794425964
accuracy_score : 0.7462222222222222

true loss : 0.5259659290313721
accuracy_score : 0.7827777777777778
"""
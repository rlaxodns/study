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


#2. 모델 구성
model = load_model("C:\\ai5\\_save\\keras49\\keras49_5_catdog.hdf5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1, batch_size=16)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test, batch_size=16)


y_pre = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)







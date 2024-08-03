# DNN -> CNN

from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D,MaxPooling2D, Flatten
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#1. 데이터 
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (581012, 54) (581012,)
# print(np.unique(y, return_counts=True))     # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],dtype=int64))
# print(pd.value_counts(y, sort=False))
# 5      9493
# 2    283301
# 1    211840
# 7     20510
# 3     35754
# 6     17367
# 4      2747

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=2321,
#                                                     stratify=y
#                                                     )

# print(x_train.shape , y_train.shape)    # (522910, 54) (522910,)
# print(x_test.shape , y_test.shape)      # (58102, 54) (58102,)


# print(pd.value_counts(y_train))
# 2    255134
# 1    190623
# 3     32172
# 7     18419
# 6     15542
# 5      8538
# 4      2482


# one hot encoding
y = pd.get_dummies(y)
# print(y.shape)  # (581012, 7)
# print(y)

# from tensorflow.keras.utils import to_categorical   # keras 이용
# y_ohe = to_categorical(y)
# print(y_ohe)
# print(y_ohe.shape)      # (581012, 8)
# y_ohe = pd.DataFrame(y_ohe)
# print(pd.value_counts(y_ohe, sort=False))


# from sklearn.preprocessing import OneHotEncoder   # sklearn 이용
# y = y.reshape(-1,1) 
# ohe = OneHotEncoder()
# y_ohe = ohe.fit_transform(y)
# print(y_ohe)
# print(y_ohe.shape)      # (581012, 7)

x = x.reshape(581012, 9, 6, 1)
x = x/255.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=5353, stratify=y)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(9,6,1), strides=1, activation='relu',padding='same')) 
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', strides=1,padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), activation='relu', strides=1, padding='same'))        
model.add(Flatten())                            

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1000,
          verbose=1, 
          validation_split=0.1,
          callbacks=[es,],
          )
end = time.time()
model.save("C:\\ai5\\_save\\keras39\\keras39_10_fetch_convtype.hdf5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],2))


y_pre = model.predict(x_test)
r2 = r2_score(y_test, y_pre)
print('r2 score :', r2)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)
print('걸린 시간 :', round(end-start, 2), '초')



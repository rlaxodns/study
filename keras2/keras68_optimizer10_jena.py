# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/code
# y는 (degC)로 잡아라
# 자르는 거 맘대로, 조건)pre = 2016.12.31 00:10부터 1.1까지 예측
# 144개

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

import tensorflow as tf
import random as rn
rn.seed(6265)
tf.random.set_seed(6265)
np.random.seed(6265)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# 데이터
data = pd.read_csv(".\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv")
data = data.drop(["Date Time"], axis=1)
print(data.shape) #(420551, 15)

x = data.head(420407)
y_pre = data.tail(144)["T (degC)"]

y = x['T (degC)']
x = x.drop(["T (degC)"], axis =1)


size = 144
def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 144):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

x = split_x(x, size)
y = split_x(y, size)

x_test1 = x[-1].reshape(-1,144,13)
# print(x)
x = np.delete(x, -1, axis =0)
y = np.delete(y, 0, axis = 0)

# y_pre = split_x(y ,size)

print(x.shape, y.shape) 
print(x_test1.shape)
# # print(y_pre)

# # x = np.delete(x, 1, axis=1)
# # y = x[1]
# # print(x.shape, y.shape) #(420264, 143, 13) (143, 13)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=231)

#2. 모델 구성
model = Sequential()
model.add(LSTM(576, input_shape = (x.shape[1], x.shape[2])))
model.add(Dense(576))
model.add(Dense(576))
model.add(Dense(288))
model.add(Dense(288))
model.add(Dense(288))
model.add(Dense(144))

#3. 컴파일 및 훈련
from tensorflow.keras.optimizers import Adam
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

from keras.callbacks import EarlyStopping, ModelCheckpoint
for i in range(0, len(lr), 1):
    model.compile(loss = 'mse', optimizer=Adam(learning_rate=lr[i]), metrics=['acc'])

    es = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience=30,
        restore_best_weights=True
    )

    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode = 'auto',
        verbose=1,
        save_best_only=True,
        filepath="C:\\ai5\\_save\\keras55\\keras55_01_.hdf5"
    )

    model.fit(x_train, y_train,
            epochs=1024,
            batch_size=1024,
            validation_split=0.2,
            callbacks=[es,mcp])

#4. 예측 및 평가
    print("===========출력==================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr:{0},로스:{1}'.format(lr[i], loss[0]))
    print('lr:{0},r2:{1}'.format(lr[i], loss[1]))
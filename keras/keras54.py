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

# 데이터
data = pd.read_csv(".\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv")
data = data.drop(["Date Time"], axis=1)

x = data.drop(['T (degC)'], axis =1)
y = data['T (degC)']

x = x[:420407] #420406

y_pre = y[420407:] #420407

print(x.shape, y.shape)

# np.save("C:\\ai5\\_data\\_save_npy\\save_keras55\\keras55_x_.npy", arr = x)
# np.save("C:\\ai5\\_data\\_save_npy\\save_keras55\\keras55_y_.npy", arr = y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
#                                                     shuffle=False)

# print(x_train.shape)

# #2. 모델 구성
# model = Sequential()
# model.add(LSTM(32, input_shape = (x.shape[1], x.shape[2])))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1))

# #3. 컴파일 및 훈련
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# model.compile(loss = 'mse', optimizer='adam')

# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'min',
#     patience=30,
#     restore_best_weights=True
# )

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode = 'auto',
#     verbose=1,
#     save_best_only=True,
#     filepath="C:\\ai5\\_save\\keras55\\keras55_01_.hdf5"
# )

# model.fit(x_train, y_train,
#           epochs=1,
#           batch_size=512,
#           validation_split=0.2,
#           callbacks=[es,mcp])

# #4. 예측 및 평가
# loss = model.evaluate(x_test, y_test)
# result = model.predict(y_pre)

# print(loss, result)
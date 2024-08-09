# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/code
# y는 (degC)로 잡아라
# 자르는 거 맘대로, 조건)pre = 2016.12.31 00:10부터 1.1까지 예측
# 144개
#jena_김태운 // 첨부파일 jena_김태운.py , h5, 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# 데이터
data = pd.read_csv(".\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv", index_col=0)

# train_dt = pd.DatetimeIndex(data.index)
# data['day'] = train_dt.day
# data['month'] = train_dt.month
# data['year'] = train_dt.year
# data['hour'] = train_dt.hour
# data['dow'] = train_dt.dayofweek

print(data.shape) #(420551, 19)

x = data.head(420407)
y_pre = data.tail(144)["T (degC)"]

print(y_pre)
y = x['T (degC)']
x = x.drop(["T (degC)"], axis =1)


size = 144

def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 1):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

x = split_x(x, size)
y = split_x(y, size)

x_test1 = x[-1].reshape(-1,144,13)
# print(x)
x = np.delete(x, -1, axis =0)
y = np.delete(y, 0, axis = 0)

print(x.shape, y.shape) 
print(x_test1.shape)
# # print(y_pre)
print(y_pre.shape)

np.save("C:\\ai5\\_data\\_save_npy\\save_keras55\\keras55_x1_.npy", arr = x)
np.save("C:\\ai5\\_data\\_save_npy\\save_keras55\\keras55_y1_.npy", arr = y)
np.save("C:\\ai5\\_data\\_save_npy\\save_keras55\\keras55_y_pre1_.npy", arr = y_pre)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4342)

#2. 모델 구성
model = Sequential()
model.add(LSTM(144, input_shape = (x.shape[1], x.shape[2]),return_sequences=True))
model.add(LSTM(144))
model.add(Dropout(0.2))
model.add(Dense(144))
model.add(Dropout(0.2))
model.add(Dense(144))

#3. 컴파일 및 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss = 'mse', optimizer='adam')

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
    filepath="C:\\ai5\\_save\\keras55\\keras55_05_.hdf5"
)

model.fit(x_train, y_train,
          epochs=1000,
          batch_size=1024,
          validation_split=0.2,
          callbacks=[es,mcp])

#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test1)
result = np.array([result]).reshape(144,1)
# acc = accuracy_score(y_pre, result)

print(loss, result)
# print(acc)
print(result.shape)
# print(y_pre)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_pre, result)

print(rmse)

#02) 2.952430614108263
#03) 1.4097586733290801
#04) 1.459469542000627
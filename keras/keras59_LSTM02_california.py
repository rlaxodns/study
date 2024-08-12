from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,
                                    random_state=6235)

##스케일링 적용##
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)

print(x_train.shape, x_test.shape) #(18576, 8) (2064, 8)
x_train = x_train.reshape(
    x_train.shape[0],
    x_train.shape[1],
    1
)
x_test = x_test.reshape(
    x_test.shape[0],
    x_test.shape[1],
    1
)
print(y_train.shape) #(18576,)

# #모델
# from keras.layers import Dropout, LSTM

# model = Sequential()

# model.add(LSTM(units = 64, input_shape = (x_train.shape[1], 1)))

# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Dense(1))

# # 컴파일 및 훈련
# import time
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(
#     monitor= 'val_loss',
#     mode = min,
#     patience=3,
#     restore_best_weights=True)

# mcp = ModelCheckpoint(
#     monitor= 'val_loss',
#     mode = 'auto',
#     patience = 10,
#     verbose= 1,
#     save_best_only=True,
#     filepath = ".//_save//keras32//keras32_dropout02_save_california.hdf5"
# )

# model.compile(loss='mse', optimizer='adam')
# st_time = time.time()
# hist = model.fit(x_train, y_train, epochs=50, batch_size=10, 
#           validation_split=0.2, verbose=0, callbacks=[es, mcp])
# end_time = time.time()

# # 예측 및 평가
# loss = model.evaluate(x_test, y_test)
# y_predict = model.predict([x_test])

# from sklearn.metrics import r2_score

# r2 = r2_score(y_test, y_predict)

# print(loss)
# print(r2)
# print(end_time-st_time)

# """
# 0.5996702909469604
# 0.561659386010252

# 0.6130666732788086
# 0.5518669095763122

# 0.6405205726623535
# 0.5317990030598638

# <스케일링 적용 후>
# 0.5360893607139587
# 0.5916927408072075
# """
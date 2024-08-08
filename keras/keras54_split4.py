# (N,10,1) -->(N, 5, 2)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM


a = np.array(range(1, 101))
x_predict = np.array(range(96,106)) # 101부터 107을 찾아라

size = 11 
print(x_predict.shape)
# x_predict = x_predict.reshape(1,10,1)

def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 1):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb.shape) #(99, 2)
x_predict = np.array(range(96,106)).reshape(1,5,2)


x = bbb[:,:-1]
y = bbb[:, -1]

print(x.shape, y.shape) #(90, 10) (90,)
x = x.reshape(90,5,2)

# # 모델 구성
# model = Sequential()
# model.add(LSTM(64, input_shape=(5, 2), activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

# # 컴파일 및 훈련
# model.compile(loss = 'mse', optimizer='adam')
# from keras.callbacks import EarlyStopping, ModelCheckpoint

# es = EarlyStopping(
#     monitor = 'loss',
#     mode = 'min',
#     patience=50, 
#     restore_best_weights=True
# )

# model.fit(x, y,
#         epochs =3,
#         batch_size=3,
#         # callbacks=[es]
#         )


# model.save("C:\\ai5\\_save\\keras54\\keras54_split3_01.hdf5")

# #4. 평가 및 예측
# loss = model.evaluate(x, y)
# result = model.predict(x_predict)

# print(loss, result)

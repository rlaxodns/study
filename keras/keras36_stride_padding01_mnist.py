#35-3 카피

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import time
from tensorflow.keras.utils import to_categorical

#1. 데이터
stt = time.time()
(x_train, y_train), (x_test, y_test) =mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) # 흑백의 경우는 1을 생략하는 경우가 있다. == (6000, 28, 28, 1)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# ####1-1 스케일링 
x_train = x_train/255.
x_test = x_test/255.

# print(np.max(x_train))
# print(np.min(x_train))

####1-2 스케일링
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test -127.5) / 127.5

# print(np.max(x_train))
# print(np.min(x_train))

#### 스케일링 MinMax, Standard Scaler
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# mms = MinMaxScaler() # 스케일러를 사용하기 위해서는 reshape를 통해 차원을 2차원으로 낮추어야 한다
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# x_train = mms.fit_transform(x_train)
# x_test = mms.transform(x_test)


# print(np.max(x_train))
# print(np.min(x_train))

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# ########to_categorical#########
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

########## OneHotEncoder ##########
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)

y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))
print(y_train.shape)

#2. 모델
from keras.layers import Dropout
model = Sequential()
model.add(Conv2D(10, (3,3),
                input_shape=(28, 28, 1),
                padding='same')) # 28X28X1  ==> 27X27X10으로 변환됨
                            # shape = (batch_size, <rows, columns, channels>)
                            # shape = (batch_size, <heights, widths, channels>)
                                                 # <input_shape부분>
model.add(Conv2D(filters = 20, kernel_size = (3, 3),
                padding='same'))
model.add(Dropout(0.2)) 
model.add(Conv2D(32, (2,2), padding='same'))

# Dense의 경우, 다양한 차원이 입력이 되지만,
# 사실상 2차원의 데이터로 변환해서 입력해야 한다.

model.add(Flatten())  # 4차원데이터를 2차원으로 변환해주는 메소드

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, input_dim=(64,)))
                        #shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto', 
    verbose=1, 
    save_best_only=True, 
    filepath=".//_save//keras35//keras35__save_cnn3_mnist01.hdf5"
)


es = EarlyStopping(
    monitor='val_loss', 
    mode = 'min',
    patience=100, 
    verbose=1, 
    restore_best_weights=True
)

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, 
        epochs = 3000, 
        batch_size = 120,
        validation_split=0.2,
        callbacks=[es, mcp])



#4. 예측 및 평가
from sklearn.metrics import accuracy_score

loss = model.evaluate(x_test, y_test)
y_pre = model.predict(x_test)
y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

print(y_pre.shape)
acc = accuracy_score(y_test, y_pre)
et = time.time()
print("loss :", loss, "acc :", acc)
print(et - stt)
# # <gpu>7.791477680206299  <cpu>63.09902763366699
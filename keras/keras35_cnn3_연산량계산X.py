import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import time

#1. 데이터
stt = time.time()
(x_train, y_train), (x_test, y_test) =mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) # 흑백의 경우는 1을 생략하는 경우가 있다. == (6000, 28, 28, 1)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28, 28, 1))) # 28X28X1  ==> 27X27X10으로 변환됨
                            # shape = (batch_size, <rows, columns, channels>)
                            # shape = (batch_size, <heights, widths, channels>)
                                                 # <input_shape부분>
model.add(Conv2D(filters = 20, kernel_size = (3, 3)))  
model.add(Conv2D(15, (4,4)))

# Dense의 경우, 다양한 차원이 입력이 되지만,
# 사실상 2차원의 데이터로 변환해서 입력해야 한다.

model.add(Flatten())  # 4차원데이터를 2차원으로 변환해주는 메소드

model.add(Dense(units=500))
model.add(Dense(units=125, input_dim=(500,)))
                        #shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', 
    mode = 'min',
    patience=1, 
    verbose=1, 
    restore_best_weights=True
)

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, 
        epochs = 30, 
        batch_size=512,
        validation_split=0.3,
        callbacks=[es])

#4. 예측 및 평가
from sklearn.metrics import accuracy_score

loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict(x_test))

acc = accuracy_score(y_test, y_pre)
et = time.time()
print("loss :", loss, "acc :", acc)
print(et - stt)
# <gpu>7.791477680206299  <cpu>63.09902763366699
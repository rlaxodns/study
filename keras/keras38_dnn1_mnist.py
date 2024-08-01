import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    #(10000, 28, 28) (10000,)

print(np.unique(y_train)) #[0 1 2 3 4 5 6 7 8 9]

##스케일링
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1, 1))

#2. 모델
model = Sequential()
model.add(Dense(128, input_shape = (28*28,)))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor= 'val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath="C:\\ai5\\_save\\keras38\\keras38_01_.hdf5"
)

model.fit(x_train, y_train,
          epochs = 100,
          batch_size= 126,
          validation_split=0.2,
          callbacks=[es, mcp])

# 평가 및 예측
loss = model.evaluate(x_test, y_test)

y_pre = np.ardmax(model.predict(x_test), axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_pre)

print(loss, acc)
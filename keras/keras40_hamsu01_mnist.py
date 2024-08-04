import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)  


##### 스케일링 1-1
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.
print(np.max(x_train), np.min(x_train))     # 1.0, 0.0

ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

#2. 모델 구성
input1 = Input(shape=(28, 28, 1 ))
dense1 = Conv2D(64, (3,3), padding='same', activation='relu')(input1)
dense2 = Conv2D(64, (3,3), padding='same', activation='relu')(dense1)
dense3 = Conv2D(64, (3,3), padding='same', activation='relu')(dense2)
drop1 = Dropout(0.2)(dense3)
dense4 = Conv2D(64, (3,3), padding='same', activation='relu')(drop1)
max = MaxPooling2D()(dense4)

flat = Flatten()(max)
dense5 = Dense(32)(flat)
drop2 = Dropout(0.3)(dense5)
dense6 = Dense(10)(drop2)
output = Dense(10, activation='softmax')(dense6)
model = Model(inputs = input1, outputs = output)

#3. 컴파일 및 훈련
model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', 
                   patience=10, verbose=1,
                   restore_best_weights=True,
                   )

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only=True,
    filepath="C:\\ai5\\_save\\keras40\\keras40_1_mnist.hdf5"
)

model.fit(x_train, y_train,
          epochs = 100, 
          batch_size=256, 
          validation_split=0.2,
          callbacks=[es, mcp])
#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_pre)
print('accuracy_score :', acc)

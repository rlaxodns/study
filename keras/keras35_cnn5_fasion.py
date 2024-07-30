import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    #(10000, 28, 28) (10000,)

# x 데이터 4차원 변환
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# y데이터 원핫인코딩
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(200, (4,4), input_shape=(28,28,1)))
model.add(Conv2D(100, (3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(100, (3,3)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 및 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=100,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    verbose=1, 
    save_best_only=True,
    filepath=".//_save//keras35//keras35__save_cnn4_mnist00.hdf5"
)

model.compile(loss = 'categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, 
          epochs=1000,
          batch_size=128,
          validation_split=0.2,
          callbacks=[es, mcp])

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
y_pre = model.predict(x_test)

y_test = y_test.to_numpy()

y_pre = np.argmax(y_pre, axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_pre)
print('acc :', acc)
print(loss,)
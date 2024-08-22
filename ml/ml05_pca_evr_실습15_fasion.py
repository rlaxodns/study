import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.datasets import mnist, fashion_mnist
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

## 스케일링
x_train = x_train/255.
y_train = y_train/255.

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)  # Should be shape (None, 10)
y_test = to_categorical(y_test, 10)    

for i in range(4):
    x = [784, 674, 459, 187]
#pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components= x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

# cum = np.cumsum(pca.explained_variance_ratio_)

# print(np.argmax(cum)+1)  # 13 # 인덱스라서 +1을 해줘야함
# print(np.argmax(cum>= 0.999)+1) # 13
# print(np.argmax(cum>= 0.99)+1)  # 12
# print(np.argmax(cum>= 0.95)+1)  # 9

    #2. 모델 구성 
    model = Sequential()
    model.add (Dense(128, input_shape=(x_train1.shape[1],)))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(128, activation='relu'))
    model.add (Dense(64, activation='relu'))
    model.add (Dense(64, activation='relu'))
    model.add (Dense(32, activation='relu'))
    model.add (Dense(32, activation='relu'))
    model.add (Dense(10, activation='softmax'))

    #3. 컴파일 및 훈련
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', 
                #   metrics=['acc']
                )

    es = EarlyStopping(
        monitor='val_loss', 
        mode = 'min',
        patience=10, 
        verbose=1, 
        restore_best_weights=True
    )
    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode = 'auto',
        verbose=1,
        save_best_only=True,
        filepath="C:\\ai5\\_save\\keras38\\keras38_02_.hdf5"
    )

    model.fit(x_train1, y_train,
            epochs=10, 
            batch_size=256,
            validation_split=0.2, 
            # callbacks=[es, mcp], 
            verbose=0)

    #4. 평가 및 예측
    loss = model.evaluate(x_test1, y_test)
    y_pre = np.argmax(model.predict(x_test1), axis=1).reshape(-1,1)
    y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)

    acc = accuracy_score(y_test1, y_pre)

    print("loss :", loss, 
        "acc :", acc)
    
"""
loss : 33648.26953125 acc : 0.1
313/313 [==============================] - 1s 2ms/step - loss: 35245.9961
loss : 35245.99609375 acc : 0.1
313/313 [==============================] - 1s 2ms/step - loss: 37096.4805
loss : 37096.48046875 acc : 0.1
313/313 [==============================] - 1s 2ms/step - loss: 26821.5020
loss : 26821.501953125 acc : 0.1
"""
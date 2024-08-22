import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))  

##### 스케일링
x_train = x_train/255.      # 0~1 사이 값으로 바뀜
x_test = x_test/255.

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

## 인코딩
y_train = to_categorical(y_train, 10)
y_test =  to_categorical(y_test, 10)

#pca
from sklearn.decomposition import PCA
pca = PCA()
x_train1 = pca.fit_transform(x_train)
x_test1 = pca.transform(x_test)

cum = np.cumsum(pca.explained_variance_ratio_)

num = [np.argmax(cum)+1,np.argmax(cum>= 0.999)+1,
       np.argmax(cum>= 0.99)+1,np.argmax(cum>= 0.95)+1 ]
for i in range(0, len(num), 1):
    pca = PCA(n_components=num[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)



    #2. 모델 구성 
    model = Sequential()
    model.add(Dense(512, input_shape=(x_train1.shape[1],)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', mode='min', 
                       patience=10, verbose=1,
                       restore_best_weights=True,
                       )
    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=1,     
        save_best_only=True,   
        filepath="C:\\ai5\\_save\\keras38\\keras38_03_.hdf5", 
    )
    model.fit(x_train1, y_train,
              epochs=100, 
              batch_size=512,
              validation_split=0.2,
            #   callbacks=[es, mcp]
              verbose=0)

    #4. 평가, 예측
    loss = model.evaluate(x_test1, y_test, verbose=1)
    print('loss :', loss[0])
    print('acc :', round(loss[1],2))

    y_pre = model.predict(x_test1)

    y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
    y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)

    r2 = accuracy_score(y_test1, y_pre)
    print('accuracy_score :', r2)
    # print("걸린 시간 :", round(end-start,2),'초')


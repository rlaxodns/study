import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import tensorflow as tf
import random as rn
rn.seed(6265)
tf.random.set_seed(6265)
np.random.seed(6265)

#1. 데이터
path1 = "C://ai5//_data//kaggle//dogs-vs-cats-redux-kernels-edition//"
sampleSubmission_csv = pd.read_csv(path1 + "sample_submission.csv", index_col=0)

   

np_path = "c:/ai5/_data/_save_npy/"
# np.save(np_path + 'keras43_01_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras43_01_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras43_01_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras43_01_y_test.npy', arr=xy_test[0][1])

x_train = np.load(np_path + 'keras43_01_x_train.npy')
y_train = np.load(np_path + 'keras43_01_y_train.npy')
x_test = np.load(np_path + 'keras43_01_x_test.npy')
y_test = np.load(np_path + 'keras43_01_y_test.npy')

x_train = x_train.reshape(
    x_train.shape[0],
    x_train.shape[1],
    x_train.shape[2]*x_train.shape[3]
)

x_test = x_test.reshape(
    x_test.shape[0],
    x_test.shape[1],
    x_test.shape[2]*x_test.shape[3]
)

print(x_train.shape, x_test.shape) #(25000, 100, 300) (12500, 100, 300)

#2. 모델 구성
from keras.layers import LSTM, Conv1D

model = Sequential()
model.add(LSTM(16, input_shape=(100,300), activation='relu', return_sequences=True)) 
model.add(Dropout(0.2))
model.add(Conv1D(16, 1, activation='relu', ))    
  
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

for i in range(0, len(lr), 1):
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr[i]), metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', 
                    patience=10, verbose=1,
                    restore_best_weights=True,
                    )

    mcp = ModelCheckpoint(
        monitor='val_loss',
        mode='auto',
        verbose=1,     
        save_best_only=True,   
        filepath="C:\\ai5\\_save\\keras49\\keras49_5_catdog.hdf5", 
    )

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=10, batch_size=16,
            validation_split=0.1,
            callbacks=[es, mcp],
            )
    end = time.time()

#4. 평가, 예측
    print("===========출력==================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr:{0},로스:{1}'.format(lr[i], loss[0]))
    print('lr:{0},r2:{1}'.format(lr[i], loss[1]))



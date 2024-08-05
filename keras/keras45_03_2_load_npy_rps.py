import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#1. 데이터
start1 = time.time()
np_path = "C:\\ai5\\_data\\_save_npy\\"
x_train = np.load(np_path + "keras45_03rps_x_train.npy")
y_train = np.load(np_path + "keras45_03rps_y_train.npy")

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=265
)

print(x_train.shape)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(100,100,3), strides=1, activation='relu',padding='same')) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu', strides=1, padding='same'))    
model.add(MaxPooling2D())    
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
    filepath='.//_save//keras45//keras45__save_brain_00.hdf5', 
)

start = time.time()
hist = model.fit(x_train, y_train,
         epochs=1000, 
         batch_size=10,
          validation_split=0.1,
          callbacks=[es, mcp],
          )
end = time.time()

#. 평가 및 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', round(loss[1],5))

y_pre = model.predict(x_test)
# r2 = r2_score(y_test,y_pre)
# print('r2 score :', r2)
print("걸린 시간 :", round(end-start,2),'초')

y_pre = np.round(y_pre)
r2 = accuracy_score(y_test, y_pre)
print('accuracy_score :', r2)
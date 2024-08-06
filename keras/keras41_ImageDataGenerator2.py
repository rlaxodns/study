import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1,
    # shear_range=0.7,
    # fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = "C:\\ai5\\_data\\image\\brain\\train"
path_test = "C:\\ai5\\_data\\image\\brain\\test"

xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200, 200), 
    batch_size=200,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200,200), 
    batch_size=200,
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True  # 테스트 데이터의 셔플은 하지 않는다.
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) #(160, 200, 200, 1) (160,)
print(x_test.shape, y_test.shape)   #(100, 200, 200, 1) (100,)


#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(200, 200, 1), strides=1, padding='same', activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 및 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=20,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto', 
    verbose=1, 
    save_best_only=True,
    filepath="C:\\ai5\\_save\\keras41\\keras41_2_imagedatagen.hdf5"
)

model.fit(x_train, y_train,
          epochs=100,
          batch_size=256, 
          validation_split=0.3,
          callbacks=[es, mcp])

#4. 모델 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pre)

print(loss, acc)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True, 
    # width_shift_range=0.5,
    # height_shift_range=0.5, 
    # rotation_range=25, 
    # # zoom_range=0.1,
    # # shear_range = 0.5,
    # fill_mode='nearest'
)

# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)

augment_size = 10000

randidx = np.random.randint(x_train.shape[0], size = augment_size)
# print(randidx)
# print(np.min(randidx), np.max(randidx)) #3 49999

x_aug = x_train[randidx].copy()
y_aug = y_train[randidx].copy()

print(x_aug.shape) #(50000, 32, 32, 3)

x_aug = train_datagen.flow(
    x_aug,
    y_aug,
    batch_size=augment_size,
    shuffle=False
).next()[0]
# print(x_aug.shape)

# print(x_train.shape)

x_train1 = np.concatenate((x_train, x_aug))
y_train = np.concatenate((y_train, y_aug))

print(x_train.shape, y_train.shape)

#인코딩#
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, BatchNormalization, Flatten
model = Sequential()
model.add(Conv2D(32, kernel_size=(2,2), input_shape = (32,32,3), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(16, (2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(254, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 및 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
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
    filepath="C:\\ai5\\_save\\keras49\\keras49_3_cifar10.hdf5"
)

model.fit(x_train, y_train,
          epochs=1000,
          batch_size=512,
          validation_split=0.2,
          callbacks=[es, mcp])

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.argmax(model.predict(x_test), axis=1).reshape(-1, 1)
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_pre)

print(loss[0], acc)
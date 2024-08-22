import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#1. 데이터 구성
train_datagen = ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    # horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    # width_shift_range=0.1,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    # rotation_range=5,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    # fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

path = "C:\\ai5\\_data\\image\\horse_human"

xy_train = train_datagen.flow_from_directory(
    path,
    target_size=(100,100),
    batch_size=2000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

x_train, x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1],
                                                    test_size=0.2, random_state=256)

##데이터 증폭## 
augment_size = 1000

randidx = np.random.randint(x_train.shape[0], size = augment_size)
print(np.min(randidx), np.max(randidx))

x_aug = x_train[randidx].copy()
y_aug = y_train[randidx].copy()

print(x_aug.shape, y_aug.shape) # (1000, 100, 100, 3) (1000,)

x_aug = train_datagen.flow(
    x_aug, y_aug,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

x_train = np.concatenate((x_train, x_aug))
y_train = np.concatenate((y_train, y_aug))

x_train = x_train.reshape(
    x_train.shape[0],
    x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
)

x_test = x_test.reshape(
    x_test.shape[0],
    x_test.shape[1]*x_test.shape[2]*x_test.shape[3]
)

print(x_train.shape, x_test.shape) #(1821, 100, 300) (206, 100, 300)

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

    #2. 모델구성
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Flatten, LSTM, Conv1D
    model = Sequential()
    model.add(Conv1D(32, 3, input_shape = (x_train1.shape[1],), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(254, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일 및 훈련
    from keras.callbacks import EarlyStopping, ModelCheckpoint
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
        filepath="C:\\ai5\\_save\\keras49\\keras49_7_horse.hdf5"
    )

    model.fit(x_train1, y_train,
            epochs=100,
            batch_size=512,
            validation_split=0.2,
            callbacks=[es, mcp])

    #4. 평가 및 예측
    loss = model.evaluate(x_test1, y_test)
    y_pre = np.round(model.predict(x_test1))
    # y_test1 = np.round(y_test)

    # acc = accuracy_score(y_test1, y_pre)

    print(loss[0], loss[1])


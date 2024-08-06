# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

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

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


#1. 데이터
path1 = "C:/ai5/_data/kaggle/dogs-vs-cats-redux-kernels-edition/"
sampleSubmission_csv = pd.read_csv(path1 + "sample_submission.csv", index_col=0)

start1 = time.time()
train_datagen = ImageDataGenerator(
    rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    # width_shift_range=0.1,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    # rotation_range=5,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    # fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)
test_datagen = ImageDataGenerator(
    rescale=1./255,              # test 데이터는 수치화만!! 
)

path_train1 = "C:\\ai5\\_data\\image\\cat_and_dog\\Train"
path_test1 = "C:\\ai5\\_data\\image\\cat_and_dog\\Test"

xy_train1 = train_datagen.flow_from_directory(
    path_train1,
    target_size=(100,100),
    batch_size=30000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

path_train2 = "C:\\ai5\\_data\\kaggle\\dogs-vs-cats-redux-kernels-edition\\train"
path_test2 = "C:\\ai5\\_data\kaggle\\dogs-vs-cats-redux-kernels-edition\\test"

xy_train2 = train_datagen.flow_from_directory(
    path_train2,
    target_size=(100, 100),
    batch_size=30000,
    class_mode='binary',
    color_mode='rgb',
    shuffle = True,
)

xy_test2 = test_datagen.flow_from_directory(
    path_test2,
    target_size=(100, 100),
    batch_size=30000,
    class_mode='binary',
    color_mode='rgb',
    shuffle = False
)

x = np.concatenate((xy_train1[0][0], xy_train2[0][0]))
y =np.concatenate((xy_train1[0][1], xy_train2[0][1]))

x_test0 = xy_test2[0][0]
y_test0 = xy_test2[0][1]

x_train, x_test, y_train, y_test = train_test_split(x, y,
                       test_size=0.2, random_state=6546)


print(x_train.shape)
aug_size = 5000
randidx = np.random.randint(x_train.shape[0], size = aug_size)

x_aug = x_train[randidx].copy()
y_aug = y_train[randidx].copy()

x_aug = x_aug.reshape(
    x_aug.shape[0],
    x_aug.shape[1],
    x_aug.shape[2], 3)

x_aug = train_datagen.flow(
    x_aug, y_aug,
    batch_size=aug_size,
    shuffle=False,
    save_to_dir="C:\\ai5\\_data\\_save_img\\05_cat_dog"
).next()[0]
x_train = np.concatenate((x_train, x_aug))
y_train = np.concatenate((y_train, y_aug))
print(x_train.shape, y_train.shape)

np.save("C:\\ai5\\_data\\_save_npy\\save_cat_dog\\x_train.npy", arr = x_train)
np.save("C:\\ai5\\_data\\_save_npy\\save_cat_dog\\y_train.npy", arr = y_train)
np.save("C:\\ai5\\_data\\_save_npy\\save_cat_dog\\x_test.npy", arr = x_test)
np.save("C:\\ai5\\_data\\_save_npy\\save_cat_dog\\y_test.npy", arr = y_test)
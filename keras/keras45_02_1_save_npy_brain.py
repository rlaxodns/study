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
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
    )
test_datagen = ImageDataGenerator(
    rescale=1./255
)

path_train = 'C:\\ai5\\_data\\image\\brain\\train\\'
path_test = 'C:\\ai5\\_data\\image\\brain\\test\\'
xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(200,200),
    batch_size=20000,
    class_mode = 'binary',
    color_mode='grayscale',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200,200),
    batch_size=20000,
    class_mode = 'binary',
    color_mode='grayscale',
    shuffle=True
)

np_path = "C:\\ai5\\_data\\_save_npy\\"
np.save(np_path + "keras45_02brain_x_train.npy", arr = xy_train[0][0])
np.save(np_path + "keras45_02brain_y_train.npy", arr = xy_train[0][1])
np.save(np_path + "keras45_02brain_x_test.npy", arr = xy_train[0][0])
np.save(np_path + "keras45_02brain_y_test.npy", arr = xy_train[0][1])

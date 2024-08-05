import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

me = np.load("C:\\ai5\\_data\\image\\me\\keras46_me_arr.npy")

model = load_model("C:\\ai5\\_save\\keras42\\k42_0804_1939_0235-0.0000.hdf5")

#평가 및 예측
result = model.predict(me)

print(1-result, "%의 확률로 ")

if np.round(result) == 0:
    print("cat")

else: print("dog")


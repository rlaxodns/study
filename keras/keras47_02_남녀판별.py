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

y_test = np.load("C:\\ai5\\_data\\_save_npy\\ke4507_gender_y_train.npy")
me = np.load("C:\\ai5\\_data\\image\\me\\keras46_me_arr.npy")



model = load_model("C:\\ai5\\_save\\keras45\\keras4507_0001.hdf5")

#평가 및 예측
result = model.predict(me)

print(result)
print(type(y_test))
result = model.predict(me)
print(np.round(result,2), "% 의 확률로")
# {'man': 0, 'woman': 1}을 기준으로 예측된 성별을 확인

if np.round(result) == 0:
    print("Predicted: man", )
else:
    print("Predicted: woman")


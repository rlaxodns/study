import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

학생csv = 'jena_김태운.csv'

path1 = 'c:\\ai5\\_data\\kaggle\\jena\\'
path2 = 'c:\\ai5\\_save\\keras55\\'

datasets = pd.read_csv(path1 + 'jena_climate_2009_2016.csv', index_col=0)
print(datasets)
print(datasets.shape)

y_정답 = datasets.iloc[-144:,1]
print(y_정답)
print(y_정답.shape)

학생꺼 = pd.read_csv(path2 + 학생csv, index_col=0)
print(학생꺼)

print(y_정답[:5])
print(학생꺼[:5])
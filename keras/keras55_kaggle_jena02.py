import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import LabelEncoder

path1 = "C:/ai5/_data/kaggle/jena/"
datasets = pd.read_csv(path1 + "jena_climate_2009_2016.csv", index_col=0)

print(datasets.shape)   # (420551, 14)

a = datasets[:-144]
print(a.shape)      # (420407, 14)

y_cor = datasets[-144:]['T (degC)']            # 예측치 정답
print(y_cor.shape)    # (144,)


# """
x_data = datasets[:-144].drop(['T (degC)'], axis=1)
y_data = datasets[:-144]['T (degC)']

print(x_data)       # [420407 rows x 13 columns]
print(y_data)       # Name: T (degC), Length: 420407, dtype: float64

size_x = 288 
size_y = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

x = split_x(x_data, size_x)
y = split_x(y_data, size_y)

x = x[:-1]
y = y[size_x-size_y+1:]

x_predict = x[-1]

print(x.shape)          # (419687, 720, 13)
print(y.shape)          # (419687, 144)
# print(x.shape, y.shape)     # (419687, 720, 13) (420263, 144)
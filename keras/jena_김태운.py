# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/code
# y는 (degC)로 잡아라
# 자르는 거 맘대로, 조건)pre = 2016.12.31 00:10부터 1.1까지 예측
# 144개

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import LabelEncoder

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# 데이터
data = pd.read_csv("C:\\ai5\\_data\\kaggle\\jena\\jena_climate_2009_2016.csv")
jena_김태운csv = data[["Date Time","T (degC)"]].tail(144)
print(jena_김태운csv) #[144 rows x 2 columns]
 
data = data.drop(["Date Time"], axis=1)
print(data.shape) #(420551, 15)


x = data.head(420407)

y_pre = data.tail(144)["T (degC)"]

y = x['T (degC)']
x = x.drop(["T (degC)"], axis =1)
print(x.shape, y.shape, y_pre.shape) #(420407, 13) (420407,) (144,)

##스케일링###
# from sklearn.preprocessing import StandardScaler
# std = StandardScaler()
# x = std.fit_transform(x)
# y = std.transform(y)
# y_pre = std.transform(y_pre)

size = 144
def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 1):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

x = split_x(x, size)
y = split_x(y, size)

x_test1 = x[-1].reshape(-1,144,13)
# print(x)
x = np.delete(x, -1, axis =0)
y = np.delete(y, 0, axis = 0)

# y_pre = split_x(y ,size)

print(x.shape, y.shape) 
print(x_test1.shape)

# x = np.delete(x, 1, axis=1)
# y = x[1]
# print(x.shape, y.shape) #(420264, 143, 13) (143, 13)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4343)



#2. 모델 구성
model = load_model("C:\\ai5\\_save\\keras55\\jena_김태운.hdf5")

#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
result = model.predict(x_test1)
result = np.array([result]).reshape(144,1)
# acc = accuracy_score(y_pre, result)

print(loss, result)
# print(acc)
print(result.shape)
# print(y_pre)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_pre, result)

print("RMSE:", rmse)

jena_김태운csv['T (degC)'] = result
jena_김태운csv.to_csv("C:\\ai5\\_save\\keras55\\jena_김태운.csv", index = False)


#RMSE: 1.3724613052063963
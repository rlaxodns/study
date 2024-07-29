from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder

# 데이터 구성
data = fetch_covtype()
x = data.data
y = data.target
# print(x.shape) #(581012, 54)
# print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

y = pd.get_dummies(y)



# ohe = OneHotEncoder(sparse=False)
# y=y.reshape(-1,1)
# y = ohe.fit_transform(y)
# print(y.shape)
# print(y)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y.shape)
# print(y)
# print(type(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=4345, stratify=y) # 분류에서만 쓰는 파라미터, 
# y라벨의 데이터를 정확하게 비율대로 나누어 준다.

print(x_train.shape, y_train.shape) #(464809, 54) (464809,)
print(x_test.shape, y_test.shape) #(116203, 54) (116203,)
# print(pd.value_counts(y_train))
# 2    226647
# 1    169475
# 3     28619
# 7     16414
# 6     13853
# 5      7622
# 4      2179

###스케일링 적용###
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)



# 모델 구성
from keras.models import load_model
model = load_model("C:\\ai5\\_save\\mcp2\\keras30_12_save_kaggle_standard.hdf5")

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pre)
print(loss, acc)

# [0.13793563842773438, 0.9503799676895142] 0.9499754739550614
# [0.23125745356082916, 0.9136000275611877] 0.9136
# [0.1510281264781952, 0.9443387985229492] 0.9436933641988589
# [0.13430580496788025, 0.9536586999893188] 0.9533058526888291
# 
from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder
import time as t
# 데이터 구성
stt = t.time()
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
from keras.layers import Dropout, Input
from keras.models import Model
# model = Sequential()
# model.add(Dense(64, input_dim = 54, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(7, activation='softmax'))

input1 = Input(shape=(54,))
dense1 = Dense(64)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 =Dense(64)(drop1)
dense3 = Dense(64)(dense2)
dense4 = Dense(64)(dense3)
dense5 = Dense(64)(dense4)
dense6 = Dense(64)(dense5)
drop2 = Dropout(0.2)(dense6)
dense7 = Dense(64)(drop2)
dense8 = Dense(64)(dense7)
dense9 = Dense(64)(dense8)
output1 = Dense(7, activation='softmax')(dense9)
model = Model(inputs = input1, outputs = output1)


# 컴파일 및 훈련
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor = 'val_loss', 
    mode = 'auto', 
    patience = 50, 
    verbose=1, 
    save_best_only=True, 
    filepath=".//_save//keras32//keras32_dropout11_save_fetch_covtype.hdf5"
)

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min', 
    patience=100 , 
    restore_best_weights= True
)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, batch_size=1024,
          validation_split=0.2, callbacks=[es, mcp])

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pre)
et = t.time()

print(loss, acc)
print(et-stt)
# [0.13793563842773438, 0.9503799676895142] 0.9499754739550614
# [0.23125745356082916, 0.9136000275611877] 0.9136
# [0.1510281264781952, 0.9443387985229492] 0.9436933641988589
# [0.13430580496788025, 0.9536586999893188] 0.9533058526888291
# <cpu>206.23246359825134 <gpu>454.09091329574585
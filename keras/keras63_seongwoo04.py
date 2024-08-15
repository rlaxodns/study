import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Flatten, BatchNormalization, MaxPool1D
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

#데이터 구성-----------------------------------------------------------------------------------------------------------------------------------------------------------------
seongwoo = pd.read_csv("C:\\ai5\\_data\\_중간고사\\성우하이텍 240816.csv", index_col=0, thousands=',')
kospi = pd.read_csv("C:\\ai5\\_data\\_중간고사\\코스피지수 과거 데이터.csv", index_col=0, thousands=',')
nasdaq = pd.read_csv("C:\\ai5\\_data\\_중간고사\\나스닥종합지수 과거 데이터 (1).csv", index_col=0, thousands=',')

# print(kospi.shape, nasdaq.shape) #(1143, 6) (1164, 6)

kospi = kospi[:948]
nasdaq = nasdaq[:948]
seongwoo = seongwoo[:948]
y_pred = seongwoo["종가"].to_numpy()
# print(seongwoo.info())
## 날짜 데이터 컬럼화------------------------------------------------------------------------------------------------------
train_dt = pd.DatetimeIndex(kospi.index)
kospi['year'] = train_dt.year
kospi['month'] = train_dt.month
kospi['day'] = train_dt.day

train_dt = pd.DatetimeIndex(nasdaq.index)
nasdaq['year'] = train_dt.year
nasdaq['month'] = train_dt.month
nasdaq['day'] = train_dt.day

train_dt = pd.DatetimeIndex(seongwoo.index)
seongwoo['year'] = train_dt.year
seongwoo['month'] = train_dt.month
seongwoo['day'] = train_dt.day

# print(kospi.info(), nasdaq.info())

kospi = kospi.drop(["거래량", "변동 %"], axis=1).to_numpy()
nasdaq = nasdaq.drop(["거래량", "변동 %"], axis=1).to_numpy()

# kospi = kospi.apply(pd.to_numeric, errors = 'coerce')
# nasdaq = nasdaq.apply(pd.to_numeric, errors = 'coerce')
# seongwoo = seongwoo.apply(pd.to_numeric, errors = 'coerce')
# print(kospi.info(), nasdaq.info())
print(kospi.shape, nasdaq.shape, y_pred.shape) #(948, 7) (948, 7)
y_pred = y_pred.reshape(948, 1)
# #스케일링-----------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
kospi = std.fit_transform(kospi)
nasdaq = std.fit_transform(nasdaq)


## 데이터 스플릿-----------------------------------------------------------------------------------------------------------------------------------
size = 3

def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 1):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

x1 = split_x(kospi, size)
x2 = split_x(nasdaq, size)
y = split_x(y_pred, size)

#훈련할 데이터 정리
x1_p = x1[:-1, :]
x2_p = x2[:-1, :]
y_p = y[1:,:]

#예측할 데이터
x1_pre = x1[-1:,:]
x2_pre = x2[-1:,:] 

print(x1_pre.shape, x2_pre.shape)
x1_pre = x1_pre.reshape(3,7)
x2_pre = x2_pre.reshape(3,7)

x1_pre = std.transform(x1_pre)
x2_pre = std.transform(x2_pre)

x1_pre = x1_pre.reshape(1,3,7)
x2_pre = x2_pre.reshape(1,3,7)


# train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_p, x2_p, y_p, test_size=0.2,random_state=4343)

print(x1_train.shape, x2_train.shape) #(756, 3, 8) (756, 3, 8)


#2-1 x1-y 모델구성
input1 = Input(shape=(3, 7 ))
c1 = Conv1D(32, 1, activation='relu')(input1)
# c1 = MaxPool1D()(c1)
# c1 = Dropout(0.5)(c1)
c1 = BatchNormalization()(c1)

c1 = Conv1D(64, 1, activation='relu')(c1)
c1 = Dropout(0.2)(c1)
c1 = BatchNormalization()(c1)

c1 = Conv1D(16, 1, activation='relu')(c1)
# c1 = Dropout(0.2)(c1)
c1= BatchNormalization()(c1)
c1 = Bidirectional(LSTM(16))(c1)

c1 = Dense(64, activation='relu')(c1)
c1 = Dense(32, activation='relu')(c1)
output1 = Dense(16)(c1)


#2-2 x2-y 모델구성
input2 = Input(shape=(3, 7))
c2 = Conv1D(32, 1, activation='relu')(input2)
# c2 = MaxPool1D()( c2)
# c2 = Dropout(0.5)(c2)
c2 = BatchNormalization()(c2)

c2 = Conv1D(64, 1, activation='relu')(c2)
c2 = Dropout(0.2)(c2)
c2 = BatchNormalization()(c2)

c2 = Conv1D(16, 1, activation='relu')(c2)
c2 = Dropout(0.2)(c2)
c2= BatchNormalization()(c2)
c2 = Bidirectional(LSTM(128))(c2)

c2 = Dense(128, activation='relu')(c2)
c2 = Dense(64, activation='relu')(c2)
output2 = Dense(32)(c2)

from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
merge = Dense(128, activation='relu')(merge)
# merge = Dropout(0.2)(merge)
merge = BatchNormalization()(merge)
merge = Dense(64, activation='relu')(merge)
merge = Dropout(0.2)(merge)
merge = Dense(32, activation='relu')(merge)
# merge = Dropout(0.2)(merge)
merge = Dense(1)(merge)

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=100,
    restore_best_weights=True
)

model = Model(inputs = [input1, input2], outputs = [merge])

model.compile(loss = "mse", optimizer='adam', metrics=['acc'])
model.fit([x1_train, x2_train], y_train,
          epochs=1000,
          batch_size=1,
          validation_split=0.3,
          callbacks=[es])

#4. 예측 및 평가
loss = model.evaluate([x1_test, x2_test], y_test)
result = model.predict([x1_pre, x2_pre])

print("loss",loss,'\n',"result", result)
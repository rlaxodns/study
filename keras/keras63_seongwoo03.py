import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Flatten
from keras.layers import Dropout, BatchNormalization, MaxPool1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터 구성
naver = pd.read_csv("C:\\ai5\\_data\\_중간고사\\NAVER 240816.csv", index_col=0, thousands=',')
seongwoo = pd.read_csv("C:\\ai5\\_data\\_중간고사\\성우하이텍 240816.csv", index_col=0, thousands=',')
hybe = pd.read_csv("C:\\ai5\\_data\\_중간고사\\하이브 240816.csv", index_col=0, thousands=',')
kia1 = pd.read_csv("C:\\ai5\\_data\\_중간고사\\기아(1)_5545_20240816.csv", index_col=0, thousands=',')
kia2 = pd.read_csv("C:\\ai5\\_data\\_중간고사\\기아(2)_5609_20240816.csv", index_col=0, thousands=',')
hyun_mo1 = pd.read_csv("C:\\ai5\\_data\\_중간고사\\현대모비스(1)_5213_20240816.csv", index_col=0, thousands=',')
hyun_mo2 = pd.read_csv("C:\\ai5\\_data\\_중간고사\\현대모비스(2)_5233_20240816.csv", index_col=0, thousands=',')
hyun_car1 = pd.read_csv("C:\\ai5\\_data\\_중간고사\\현대차(1)_5444_20240816.csv", index_col=0, thousands=',')
hyun_car2 = pd.read_csv("C:\\ai5\\_data\\_중간고사\\현대차(2)_5506_20240816.csv", index_col=0, thousands=',')
seongwoo = pd.read_csv("C:\\ai5\\_data\\_중간고사\\성우하이텍 240816.csv", index_col=0, thousands=',')

kia = pd.concat([kia1, kia2])
hyun_mo = pd.concat([hyun_mo1, hyun_mo2])
hyun_car = pd.concat([hyun_car1, hyun_car2])

# print(kia.shape, hyun_mo.shape, hyun_car.shape) #(986, 10) (986, 10) (986, 10)

kia = kia[:948]
hyun_mo = hyun_mo[:948]
hyun_car = hyun_car[:948]
seongwoo = seongwoo[:948]
naver = naver[:948]
seongwoo = seongwoo[:948]


kia = kia.drop(["대비","거래대금", "시가총액", "상장주식수", "등락률"], axis=1)
hyun_mo = hyun_mo.drop(["대비","거래대금", "시가총액", "상장주식수", "등락률"], axis=1)
hyun_car = hyun_car.drop(["대비","거래대금", "시가총액", "상장주식수", "등락률"], axis=1)

## 라벨 인코딩.
le = LabelEncoder()
naver["전일비"] = le.fit_transform(naver["전일비"])
seongwoo["전일비"] = le.fit_transform(seongwoo["전일비"])
hybe["전일비"] = le.fit_transform(hybe["전일비"])

# print(kia.shape, hyun_mo.shape, hyun_car.shape, seongwoo.shape, naver.shape, hybe.shape) #(948, 5) (948, 5) (948, 5) (948, 5) (948, 5) (948, 5)

## 날짜 데이터 컬럼화
train_dt = pd.DatetimeIndex(naver.index)
naver['year'] = train_dt.year
naver['month'] = train_dt.month
naver['day'] = train_dt.day

train_dt = pd.DatetimeIndex(hybe.index)
hybe['year'] = train_dt.year
hybe['month'] = train_dt.month
hybe['day'] = train_dt.day

train_dt = pd.DatetimeIndex(seongwoo.index)
seongwoo['year'] = train_dt.year
seongwoo['month'] = train_dt.month
seongwoo['day'] = train_dt.day

train_dt = pd.DatetimeIndex(kia.index)
kia['year'] = train_dt.year
kia['month'] = train_dt.month
kia['day'] = train_dt.day

train_dt = pd.DatetimeIndex(hyun_mo.index)
hyun_mo['year'] = train_dt.year
hyun_mo['month'] = train_dt.month
hyun_mo['day'] = train_dt.day

train_dt = pd.DatetimeIndex(hyun_car.index)
hyun_car['year'] = train_dt.year
hyun_car['month'] = train_dt.month
hyun_car['day'] = train_dt.day

##데이터 형식 변경
naver = naver.apply(pd.to_numeric, errors = 'coerce')
seongwoo = seongwoo.apply(pd.to_numeric, errors = 'coerce')
hybe = hybe.apply(pd.to_numeric, errors = 'coerce')
kia = kia.astype(float)
hyun_mo = hyun_mo.astype(float)
hyun_car = hyun_car.astype(float)

###_____________________________________________________________________________________________________________________________
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
naver = std.fit_transform(naver)
hybe = std.fit_transform(hybe)
kia = std.fit_transform(kia)
hyun_mo = std.fit_transform(hyun_mo)
hyun_car = std.fit_transform(hyun_car)

## 예측할 y_data 구성
y_pred = seongwoo["종가"]

## 데이터 스플릿
size = 3

def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 1):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

x1 = split_x(naver, size)
x2 = split_x(hybe, size)
x3 = split_x(kia, size)
x4 = split_x(hyun_mo, size)
x5 = split_x(hyun_car, size)
y = split_x(y_pred, size)

## 훈련할 데이터 정리
x1_p = x1[:-1, :]
x2_p = x2[:-1, :]
x3_p = x3[:-1, :]
x4_p = x4[:-1, :]
x5_p = x5[:-1, :]
y_p = y[1:,:]

### 예측data
x1_pre = x1[-1:,:]
x2_pre = x2[-1:,:] 
x3_pre = x3[-1:,:] 
x4_pre = x4[-1:,:] 
x5_pre = x5[-1:,:] 

# print(x1_pre.shape, x2_pre.shape, x3_pre.shape, x4_pre.shape) #(1, 3, 19) (1, 3, 19) (1, 3, 8) (1, 3, 8)

##reshape
x1_pre = x1_pre.reshape(3,19)
x2_pre = x2_pre.reshape(3,19)
x3_pre = x3_pre.reshape(3,8)
x4_pre = x4_pre.reshape(3,8)
x5_pre = x5_pre.reshape(3,8)

x1_pre = std.transform(x1_pre)
x2_pre = std.transform(x2_pre)
x3_pre = std.transform(x3_pre)
x4_pre = std.transform(x4_pre)
x5_pre = std.transform(x5_pre)


x1_pre = x1_pre.reshape(1,3,19)
x2_pre = x2_pre.reshape(1,3,19)
x3_pre = x3_pre.reshape(1,3,8)
x4_pre = x4_pre.reshape(1,3,8)
x5_pre = x5_pre.reshape(1,3,8)


# train_test_split
x1_train, x1_test, x2_train, x2_test,x3_train, x3_test, x4_train, x4_test, x5_train, x5_test,\
y_train, y_test = train_test_split(x1_p, x2_p,x3_p,x4_p, x5_p, y_p, test_size=0.2,random_state=4343)

# print(x1_train.shape, x2_train.shape, x3_train.shape, x4_train.shape, x5_train.shape, y_train.shape)
# (757, 1, 19) (757, 1, 19) (757, 1, 8) (757, 1, 8) (757, 1, 8) (757, 1)

#2-1 x1-y 모델구성
input1 = Input(shape=(3, 19))
c1 = Conv1D(32, 1, activation='relu')(input1)
c1 = MaxPool1D()(c1)
# c1 = Dropout(0.5)(c1)
c1 = BatchNormalization()(c1)

c1 = Conv1D(64, 1, activation='relu')(c1)
c1 = Dropout(0.2)(c1)
c1 = BatchNormalization()(c1)

c1 = Conv1D(16, 1, activation='relu')(c1)
c1 = Dropout(0.2)(c1)
c1= BatchNormalization()(c1)
c1 = Bidirectional(LSTM(16))(c1)

c1 = Dense(64, activation='relu')(c1)
c1 = Dense(32, activation='relu')(c1)
output1 = Dense(16)(c1)


#2-2 x2-y 모델구성
input2 = Input(shape=(3, 19))
c2 = Conv1D(32, 1, activation='relu')(input2)
c2 = MaxPool1D()(c2)
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

#2-3 x3-y 모델구성
input3 = Input(shape=(3, 8))
c3 = Conv1D(32, 1, activation='relu')(input3)
c3 = MaxPool1D()(c3)
# c3 = Dropout(0.5)(c3)
c3 = BatchNormalization()(c3)

c3 = Conv1D(64, 1, activation='relu')(c3)
c3 = Dropout(0.2)(c3)
c3 = BatchNormalization()(c3)

c3 = Conv1D(16, 1, activation='relu')(c3)
c3 = Dropout(0.2)(c3)
c3= BatchNormalization()(c3)
c3 = Bidirectional(LSTM(16))(c3)

c3 = Dense(128, activation='relu')(c3)
c3 = Dense(64, activation='relu')(c3)
output3 = Dense(32)(c3)

#2-4 x4-y 모델구성
input4 = Input(shape=(3, 8))
c4 = Conv1D(32, 1, activation='relu')(input4)
c4 = MaxPool1D()(c4)
# c4 = Dropout(0.5)(c4)
c4 = BatchNormalization()(c4)

c4 = Conv1D(64, 1, activation='relu')(c4)
c4 = Dropout(0.2)(c4)
c4 = BatchNormalization()(c4)

c4 = Conv1D(16, 1, activation='relu')(c4)
c4 = Dropout(0.2)(c4)
c4= BatchNormalization()(c4)
c4 = Bidirectional(LSTM(16))(c4)

c4 = Dense(128, activation='relu')(c4)
c4 = Dense(64, activation='relu')(c4)
output4 = Dense(32)(c4)

#2-5 x5-y 모델구성
input5 = Input(shape=(3, 8))
c5 = Conv1D(32, 1, activation='relu')(input5)
c5 = MaxPool1D()(c5)
# c5 = Dropout(0.5)(c5)
c5 = BatchNormalization()(c5)

c5 = Conv1D(64, 1, activation='relu')(c5)
c5 = Dropout(0.2)(c5)
c5 = BatchNormalization()(c5)

c5 = Conv1D(16, 1, activation='relu')(c5)
c5 = Dropout(0.2)(c5)
c5= BatchNormalization()(c5)
c5 = Bidirectional(LSTM(16))(c5)

c5 = Dense(128, activation='relu')(c5)
c5 = Dense(64, activation='relu')(c5)
output5 = Dense(32)(c5)


#2-6 data merge1
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

#2-6 datamerge2
merge1 = concatenate([output3, output4, output5])
merge1 = Dense(128, activation='relu')(merge1)
merge1 = Dropout(0.2)(merge1)
merge1 = BatchNormalization()(merge1)
merge1 = Dense(128, activation='relu')(merge1)
# merge1 = Dropout(0.2)(merge1)
merge1 = Dense(64, activation='relu')(merge1)
# merge1 = Dropout(0.2)(merge1)
merge1 = Dense(1)(merge1)

#2-6 datamerge3
lo = concatenate([merge, merge1])
lo = Dense(1)(lo)
model = Model(inputs = [input1, input2, input3, input4, input5], outputs = [lo])


#3. 컴파일 및 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=100,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath="C:\\ai5\\_save\\keras63\\keras63_06.h5"
)

model.compile(loss = "mse", optimizer='adam', metrics=['acc'])
model.fit([x1_train, x2_train, x3_train, x4_train, x5_train], y_train,
          epochs=1000,
          batch_size=2,
          validation_split=0.3,
          callbacks=[es,mcp])

#4. 예측 및 평가
loss = model.evaluate([x1_test, x2_test, x3_test, x4_test, x5_test], y_test)
result = model.predict([x1_pre, x2_pre, x3_pre, x4_pre, x5_pre])

print("loss",loss,'\n',"result", result)

""" 
loss [4765899.5, 0.0] 
result
[[6747.7363]]

loss [4927744.0, 0.0] 
result
[[6906.614]]

03)
loss [4923399.0, 0.0] 
 result [[7167.72]]

04)
loss [5039714.5, 0.0] 
 result [[7144.0264]] 






 
 """
import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터 구성
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

kia = kia.drop(["대비","거래대금", "시가총액", "상장주식수", "등락률"], axis=1)
hyun_mo = hyun_mo.drop(["대비","거래대금", "시가총액", "상장주식수", "등락률"], axis=1)
hyun_car = hyun_car.drop(["대비","거래대금", "시가총액", "상장주식수", "등락률"], axis=1)
seongwoo = seongwoo.drop(["전일비", "Unnamed: 6", "등락률", "금액(백만)", "신용비","개인","기관","외인(수량)","외국계","프로그램","외인비"], axis=1)

# print(kia.shape, hyun_mo.shape, hyun_car.shape, seongwoo.shape,
#       kia.info(), seongwoo.info())
seongwoo = seongwoo.astype(float)
kia = kia.astype(float)
hyun_mo = hyun_mo.astype(float)
hyun_car = hyun_car.astype(float)

# print(kia.info())

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

train_dt = pd.DatetimeIndex(seongwoo.index)
seongwoo['year'] = train_dt.year
seongwoo['month'] = train_dt.month
seongwoo['day'] = train_dt.day

# print(kia.shape, hyun_mo.shape, hyun_car.shape, seongwoo.shape,) #(948, 8) (948, 8) (948, 8) (948, 8)

## 예측할 y_data 구성
y_pred = seongwoo["종가"]
# print(y_pred.shape) #(948,)

## 데이터 스플릿
size = 1

def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 1):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

x1 = split_x(kia, size)
x2 = split_x(hyun_mo, size)
x3 = split_x(hyun_car, size)
y = split_x(y_pred, size)

###훈련data
x1_p = x1[:-1, :]
x2_p = x2[:-1, :]
x3_p = x3[:-1, :]
y_p = y[1:,:]

### 예측data
x1_pre = x1[-1:,:]
x2_pre = x2[-1:,:] 
x3_pre = x3[-1:,:] 

x1_train, x1_test, x2_train, x2_test,x3_train, x3_test ,y_train, y_test = train_test_split(x1_p, x2_p, x3_p,y_p,
                                                                         test_size=0.2, 
                                                                         random_state=4343)

print(x2_train.shape) #(757, 1, 8)

#2-1 x1-y 모델구성
input1 = Input(shape=(1, 8))
conv1_1 = Conv1D(filters=512, kernel_size=1)(input1)
bi_lstm1 = Bidirectional(LSTM(256, activation='relu'))(conv1_1)
dense1_1 = Dense(256, activation='relu')(bi_lstm1)
dense1_2 = Dense(128, activation='relu')(dense1_1)
output1 = Dense(64)(dense1_2)

#2-2 x2-y 모델구성
input2 = Input(shape=(1, 8))
conv1_2 = Conv1D(filters=512, kernel_size=1)(input2)
bi_lstm2 = Bidirectional(LSTM(256, activation='relu'))(conv1_2)
dense2_1 = Dense(256, activation='relu')(bi_lstm2)
dense2_2 = Dense(128, activation='relu')(dense2_1)
output2 = Dense(64)(dense2_2)

#2-3 x1-y 모델구성
input3 = Input(shape=(1, 8))
conv1_3 = Conv1D(filters=512, kernel_size=1)(input3)
bi_lstm3 = Bidirectional(LSTM(256, activation='relu'))(conv1_3)
dense3_1 = Dense(256, activation='relu')(bi_lstm3)
dense3_2 = Dense(128, activation='relu')(dense3_1)
output3 = Dense(64)(dense3_2)

#2-3 data merge
from keras.layers.merge import concatenate
merge = concatenate([output1, output2, output3])
merge2 = Dense(256, activation='relu')(merge)
merge3 = Dense(1)(merge2)

model = Model(inputs = [input1, input2, input3], outputs = merge3)


#3. 컴파일 및 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=10,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    save_best_only=True,
    filepath="C:\\ai5\\_save\\keras63\\keras63_01.h5"
)

model.compile(loss = "mse", optimizer='adam')
model.fit([x1_train, x2_train, x3_train], y_train,
          epochs=1000,
          batch_size=32,
          validation_split=0.3,
          callbacks=[es,]
        #  mcp
        # ]
          )

#4. 예측 및 평가
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
result = model.predict([x1_pre, x2_pre, x3_pre])

print("loss",loss,'\n',"result",'\n', result)
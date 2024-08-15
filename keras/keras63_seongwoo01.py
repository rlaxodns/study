import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional
from keras.layers import MaxPooling1D, BatchNormalization, Flatten, MaxPool1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터 구성
naver = pd.read_csv("C:\\ai5\\_data\\중간고사데이터\\NAVER 240816.csv", index_col=0, thousands=',')
seongwoo = pd.read_csv("C:\\ai5\\_data\\중간고사데이터\\성우하이텍 240816.csv", index_col=0, thousands=',')
hybe = pd.read_csv("C:\\ai5\\_data\\중간고사데이터\\하이브 240816.csv", index_col=0, thousands=',')

# print(naver.shape, seongwoo.shape, hybe.shape) #(5390, 17) (7058, 17) (948, 17)
naver = naver[:948]
seongwoo = seongwoo[:948]

print(naver.shape, seongwoo.shape) #(948, 16) (948, 16)
# print(naver.info())
"""
 0   시가          948 non-null    object
 1   고가          948 non-null    object
 2   저가          948 non-null    object
 3   종가          948 non-null    object
 4   전일비         948 non-null    object
 5   Unnamed: 6  948 non-null    object
 6   등락률         948 non-null    float64
 7   거래량         948 non-null    object
 8   금액(백만)      948 non-null    object
 9   신용비         948 non-null    float64
 10  개인          948 non-null    object
 11  기관          948 non-null    object
 12  외인(수량)      948 non-null    object
 13  외국계         948 non-null    object
 14  프로그램        948 non-null    object
 15  외인비         948 non-null    float64"""

## 라벨 인코딩.
le = LabelEncoder()
naver["전일비"] = le.fit_transform(naver["전일비"])
seongwoo["전일비"] = le.fit_transform(seongwoo["전일비"])
hybe["전일비"] = le.fit_transform(hybe["전일비"])
# print(naver.info(), seongwoo.info(), hybe.info())

##데이터 형식 변경
naver = naver.apply(pd.to_numeric, errors = 'coerce')
seongwoo = seongwoo.apply(pd.to_numeric, errors = 'coerce')
hybe = hybe.apply(pd.to_numeric, errors = 'coerce')
print(naver.info(), seongwoo.info(), hybe.info())

# print(naver)

## 날짜 데이터 컬럼화
# train_dt = pd.DatetimeIndex(naver.index)
# naver['year'] = train_dt.year
# naver['month'] = train_dt.month
# naver['day'] = train_dt.day

# train_dt = pd.DatetimeIndex(hybe.index)
# hybe['year'] = train_dt.year
# hybe['month'] = train_dt.month
# hybe['day'] = train_dt.day

# train_dt = pd.DatetimeIndex(seongwoo.index)
# seongwoo['year'] = train_dt.year
# seongwoo['month'] = train_dt.month
# seongwoo['day'] = train_dt.day
# print(naver.shape, seongwoo.shape, hybe.shape) #(948, 19) (948, 19) (948, 19)

## 예측할 y_data 구성
y_pred = seongwoo["종가"]
# print(y_pred.shape) #(948,)

## 데이터 스플릿
size = 11

def split_x(data, size):
    aaa=[]
    for i in range(len(data) - size + 1):
        sub = data[i : (i+size)]
        aaa.append(sub)
    return np.array(aaa)

x1 = split_x(naver, size)
x2 = split_x(hybe, size)
y = split_x(y_pred, size)

# print(x1.shape, x2.shape, y.shape) #(948, 1, 19) (948, 1, 19) (948, 1)

## 훈련할 데이터 정리
x1_p = x1[:-1, :]
x2_p = x2[:-1, :]
y_p = y[1:,:]
# print(x1_p.shape, x2_p.shape, y_p.shape) #(947, 1, 19) (947, 1, 19) (947, 1)

## 예측 데이터
x1_pre = x1[-1:,:]
x2_pre = x2[-1:,:] 
# print(x1_pre.shape, x2_pre.shape) #(1, 1, 19) (1, 1, 19)

# ##reshape 사용을 위한 넘파이 변환
# x1_p_train = x1_p_train.to_numpy()
# x2_p_train = x2_p_train.to_numpy()
# y_p_train  = y_p_train.to_numpy()

# x1_pre = x1_pre.to_numpy()
# x2_pre = x2_pre.to_numpy()

# ## RNN을 사용하기 위한 Reshape
# x1_p_train = x1_p_train.reshape(947, 19, 1)
# x2_p_train = x2_p_train.reshape(947, 19, 1)
# y_p_train = y_p_train.reshape(947, 1)

# x1_pre = x1_pre.reshape(1, 19, 1)
# x2_pre = x2_pre.reshape(1, 19, 1)

# train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_p, x2_p, y_p,
                                                                         test_size=0.1, 
                                                                         random_state=321)


#2-1 x1-y 모델구성
input1 = Input(shape=(11, 16))
c1 = Conv1D(32, 1, activation='relu')(input1)
c1 = Conv1D(64, 1, activation='relu')(input1)
c1 = MaxPool1D()(c1)
c1 = BatchNormalization()(c1)
c1 = Bidirectional(LSTM(128))(c1)

c1 = Dense(64, activation='relu')(c1)
# c1 = Dense(32, activation='relu')(c1)
c1 = Dense(32, activation='relu')(c1)
output1 = Dense(16, activation='relu')(c1)


#2-2 x2-y 모델구성
input2 = Input(shape=(11, 16))
c2 = Conv1D(32, 1, activation='relu')(input1)
c2 = Conv1D(64, 1, activation='relu')(input1)
c2 = MaxPool1D()(c2)
c2 = BatchNormalization()(c2)
c2 = Bidirectional(LSTM(128))(c2)

c2 = Dense(64, activation='relu')(c2)
c2 = Dense(32, activation='relu')(c2)
output2 = Dense(16, activation='relu')(c2)

#2-3 data merge
from keras.layers.merge import concatenate
merge = concatenate([output1, output2])
merge1 = Dense(32, activation='relu')(merge)
merge1 = BatchNormalization()(merge1)
merge1 = Dense(32, activation='relu')(merge1)
merge1 = Dense(1)(merge1)

model = Model(inputs = [input1, input2], outputs = merge1)

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
    filepath="C:\\ai5\\_save\\keras63\\keras6301_06.h5"
)

model.compile(loss = "mse", optimizer='adam')
model.fit([x1_train, x2_train], y_train,
          epochs=1000,
          batch_size=2,
          validation_split=0.3,
          callbacks=[es,
         mcp
        ]
          )

#4. 예측 및 평가
loss = model.evaluate([x1_test, x2_test], y_test)
result = model.predict([x1_pre, x2_pre])

print("loss",loss,'\n',"result",'\n', result)

#  02 [[7373.165]]
# 03 [[7558.927]]
# 04  [[7579.1914]]
# 05  [[7491.947]

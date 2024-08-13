# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd #csv파일 사용시 데이터의 인덱스와 컬럼을 분리시 사용
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. data

path = "C:\\ai5\\_data\\dacon\\따릉이\\" # 상대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0) 

print(train_csv) 

test_csv = pd.read_csv(path + "test.csv", index_col=0) 
print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0) 
print(submission_csv)  
print(train_csv.shape) #(1459, 10)
print(test_csv.shape) #(715, 9)
print(submission_csv.shape) #(715, 1)

print(train_csv.columns) 
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
# 카운트를 제외해야함

print(train_csv.info()) 

print(train_csv.isna().sum()) # 각 컬럼의 데이터 결측치의 수 확인 방법

train_csv = train_csv.dropna() #데이터 내의 na값을 제외한다.
# 결측치가 있는 인덱스를 dropna()하면 행자체가 삭제됨

print(train_csv.isna().sum())
print(train_csv.info())
print(train_csv) #[1328 rows x 10 columns]

print(test_csv.info())

print(test_csv.isna().sum()) 

test_csv = test_csv.fillna(test_csv.mean())  # .fillna  : 결측치를 채우는 함수
                                             # .mean() 각 컬럼의 평균값을 채워진다.
print(test_csv.info())


#############지금까지 데이터 전처리 과정#################################################

x = train_csv.drop(['count'], axis=1)  # axis=0은 행의 [---]이름의 행 삭제, 
# axis=1는 열의 [---]이름의 열 삭제
print(x) #[1328 rows x 9 columns]

y = train_csv['count'] #y는 카운트 열만 지정
print(y.shape) # (1328,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        test_size=0.2, shuffle=True, random_state=4345)
# print(x_train.shape) # (929, 9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = mas.fit_transform(x_train)
x_test = mas.transform(x_test)

x_train = x_train.reshape(
    x_train.shape[0], 
    x_train.shape[1],
    1
)
x_test = x_test.reshape(
    x_test.shape[0],
    x_test.shape[1],
    1
)

print(x_train.shape, x_test.shape)
x_train = x_train.reshape(
    x_train.shape[0],
    x_train.shape[1],
    1
)
x_test = x_test.reshape(
    x_test.shape[0],
    x_test.shape[1],
    1
)

#2. 모델 구성
from keras.layers import Dropout, Conv1D, Flatten

model = Sequential()
model.add(Conv1D(1000,2, input_shape=(x_train.shape[1], 1)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(125))
model.add(Dropout(0.2))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 및 훈련
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto', 
    patience = 50, 
    save_best_only= True,
    verbose=1, 
    filepath= ".//_save//keras32//keras32_dropout04_save_dacon_ddareung.hdf5"
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 50,
    restore_best_weights = True
)

model.compile(loss='mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=1,
                 validation_split = 0.2, callbacks = [es, mcp])
end_time = time.time()



#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)

y_submit = model.predict([test_csv]) 
print(y_submit)
print(y_submit.shape)

y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)



print("result", y_submit)
print("loss", loss)
print("r2", r2)
print(end_time - start_time)


"""
<적용후>
loss 2536.8232421875
r2 0.5750679270190111

loss 2674.634765625
r2 0.5519837682738082
-------------------------
loss 2960.484130859375
r2 0.5492597268086088
random_state=625
epochs=200, batch_size=1

loss 2921.001708984375
r2 0.5720832778153639
random_state=625
epochs=300, batch_size=1

loss 2857.32373046875
r2 0.5561720186542514
random_state=6265

loss 3012.35205078125
r2 0.5507539231967338
random_state=6245

loss 3088.42919921875
r2 0.5560217793278813
random_state=724

loss 3151.997802734375
r2 0.4944037620881466
random_state=7245
epochs=1500

loss 2557.16357421875
r2 0.6022284477446165
"""


##submission.csv 만들기 // count컬럼 값만 넣어주면 됨
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)
submission_csv.to_csv(path+"submission_0716_1640.csv")


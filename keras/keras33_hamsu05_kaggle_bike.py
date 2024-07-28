# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv
# 현재는 결측치만 확인하고 있으나 추후 이상치 또한 확인해야함
# 

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
path = "C:/Users/kim/Downloads/bike-sharing-demand/"  #절대경로 
#C:\Users\kim\Downloads\bike-sharing-demand\  # \n과 같은 경우에는 특수문자로 인지하기에 /로 바꿔주거나 \\ 또는 //를 사용
#/ // \ \\모두 사용은 가능

train_csv = pd.read_csv(path+"train.csv", index_col=0)  # 첫 번쨰 데이터는 시간 데이터이기 때문에 분할의 필요하여 복잡하니 우선은 인덱스로 잡기
test_csv = pd.read_csv(path+"test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv", index_col=0)

print(train_csv.shape, test_csv.shape, sampleSubmission_csv.shape)  #(10886, 11) (6493, 8) (6493, 1) # train에 추가 2개의 컬럼 존재

print(train_csv.columns) #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                                   #'humidity', 'windspeed', 'casual', 'registered', 'count']
print(train_csv.info()) #결측치 없음
print(test_csv.info()) #결측치 없음

print(train_csv.describe()) # 데이터들의 평균, 중위값, 표차 등을 설명해주는 함수
"""             season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000""" 
# std:표준편차      # 평균과 중위값의 격차를 고려해야 한다
# 중위값과 데이터 중에서 크게 차이나는 값을 '이상치'라고 한다 # 1/4분위 & 3/4분위

#########결측치 확인############
print(train_csv.isna().sum())
print(test_csv.isna().sum())

## x와 y를 분리
x = train_csv.drop(['casual', 'registered','count'], axis=1)  #[]:대괄호는 파이썬의 리스트 {}: 딕셔너리 (): 튜플
# 따라서 리스트 형태이기 때문에 각각을 대괄호로 묶는 것이 아니라 전체를 괄호치면서 리스트로 만든다
# 두개 이상은 리스트 따라서 한개는 소괄호만으로도 가능
print(x.shape) #(10886, 8)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=9999)

##스케일링##
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
###########


#2. 모델
from keras.layers import Dropout, Input
from keras.models import Model

# model = Sequential()
# model.add(Dense(1000, activation = 'relu', input_dim = 8))
#  # activation function(활성화 함수)을 통해서 
#  # 음수를 배제할 수 있음, relu: 양수는 그대로, 음수는 0으로 
# model.add(Dropout(0.2))
# model.add(Dense(900, activation = 'relu'))
# model.add(Dense(800, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(470, activation = 'relu'))
# model.add(Dense(260, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(150, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(1, activation = 'linear'))

input1 = Input(shape = (8,))
dense1 = Dense(1000)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(900)(drop1)
dense3 = Dense(800)(dense2)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(470)(drop2)
dense5 = Dense(260)(dense4)
drop3 = Dropout(0.2)(dense5)
dense6 = Dense(150)(drop3)
dense7 = Dense(50)(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs = input1, outputs = output1)


#3. 컴파일 및 훈련
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor= 'val_loss', 
    mode = 'auto', 
    patience = 20, 
    verbose=1,
    save_best_only=True,
    filepath=".//_save//keras32//keras32_dropout05_save_kaggle_bike.hdf5"
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights = True
)

model.compile(loss = 'mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size = 256,
                 validation_split=0.2, verbose =2, callbacks = [es, mcp])
end_time = time.time()

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([test_csv])
y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

print("result", result)
print('loss', loss)
print('r2', r2)

# print(hist)
# print(hist.history)
# print(round(end_time - start_time, 1))
# #5. 파일 출력
# sampleSubmission_csv['count'] = result
# print(sampleSubmission_csv.shape)
# sampleSubmission_csv.to_csv(path+"sampleSubmission_0717_1530.csv")

"""
loss 24723.44140625
r2 0.23068426139316467

<적용후>
loss 21343.37109375
r2 0.33586148998866894

loss 21998.580078125
r2 0.3154733919243766
"""

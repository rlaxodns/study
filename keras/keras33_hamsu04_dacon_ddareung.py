# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd #csv파일 사용시 데이터의 인덱스와 컬럼을 분리시 사용
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. data

path = "./_data/dacon/따릉이/" # 상대경로

#경로를 지정한 변수를 통해서 편하게 경로를 지정할 수 있음
#쌍따옴표 안에서의 숫자 등은 문자로 인식
#"문자"+"문자"= 문자문자 # + 는 단순 연결시 사용

train_csv = pd.read_csv(path + "train.csv", index_col=0) 
                                                  #첫 번째 컬럼은 데이터가 아니고 인덱스라 해줘 
                       # . : 점 하나는 루트(뿌리)를 의미
# train_csv에 파일의 데이터가 입력

print(train_csv)  #[1459 rows x 11 columns]  
# index_col=0입력시, [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0) 
print(test_csv) #[715 rows x 9 columns]

submission_csv = pd.read_csv(path + "submission.csv", index_col=0) 
print(submission_csv)  #[715 rows x 1 columns] # NaN = 결측치
#결측치를 구하는 방법 ex) 선형회귀
#파일이름이 한글이라 문제가 생기는 경우에는 영문으로 바꾸면 됨

#가장 많이 틀리는것, 오타, 경로, 형태(나: input_dim = )
#오류의 해결, 첫째 라인을 찾고, 둘째 문제를 읽고, 셋째 오타인지 등등

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
#각 컬럼의 데이터의 수가 다름으로써 각각의 컬럼에 결측치가 존재
#y값에 결측치가 있는 것은 문제가 있으나, 현재의 데이터에 y에는 결측치가 없음
#x값의 결측치가 있는 것은 크게 문제 안됨
""" 0   hour                    1459 non-null   int64
 1   hour_bef_temperature    1457 non-null   float64
 1   hour_bef_temperature    1457 non-null   float64
 2   hour_bef_precipitation  1457 non-null   float64
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64
 9   count                   1459 non-null   float64
 """

# 데이터가 존재할 때, 결측치가 있는 경우 1) 결측치를 삭제(데이터가 많은 경우), 또는 2) 결측치를 채워주거나
print(train_csv.isna().sum()) # 각 컬럼의 데이터 결측치의 수 확인 방법

train_csv = train_csv.dropna() #데이터 내의 na값을 제외한다.
# 결측치가 있는 인덱스를 dropna()하면 행자체가 삭제됨

print(train_csv.isna().sum())
print(train_csv.info())
print(train_csv) #[1328 rows x 10 columns]

print(test_csv.info())
"""---  ------                  --------------  -----
 0   hour                    715 non-null    int64
 1   hour_bef_temperature    714 non-null    float64
 2   hour_bef_precipitation  714 non-null    float64
 3   hour_bef_windspeed      714 non-null    float64
 4   hour_bef_humidity       714 non-null    float64
 5   hour_bef_visibility     714 non-null    float64
 6   hour_bef_ozone          680 non-null    float64
 7   hour_bef_pm10           678 non-null    float64
 8   hour_bef_pm2.5          679 non-null    float64"""
print(test_csv.isna().sum())  # 테스트 데이터의 결측치를 제거하는 경우에는 
# 서브미션에 추가해야하는 데이터를 그냥 제거하는 방법이기 때문에 문제가 생긴다
# NaN은 공백의 데이터기 때문에 연산을 할 수 없음. 그렇기에 정확의 큰 문제가 발생한다.
# 그래서 임의의 값을 대입하는데 "평균치"를 통해서 대입할 수도 있다.

test_csv = test_csv.fillna(test_csv.mean())  # .fillna  : 결측치를 채우는 함수
                                             # .mean() 각 컬럼의 평균값을 채워진다.
print(test_csv.info())
"""---  ------                  --------------  -----
 0   hour                    715 non-null    int64
 1   hour_bef_temperature    715 non-null    float64
 2   hour_bef_precipitation  715 non-null    float64
 3   hour_bef_windspeed      715 non-null    float64
 4   hour_bef_humidity       715 non-null    float64
 5   hour_bef_visibility     715 non-null    float64
 6   hour_bef_ozone          715 non-null    float64
 7   hour_bef_pm10           715 non-null    float64
 8   hour_bef_pm2.5          715 non-null    float64"""


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

#2. 모델 구성
from keras.layers import Dropout, Input
from keras.models import Model

# model = Sequential()
# model.add(Dense(1000, input_dim=9))
# model.add(Dropout(0.2))
# model.add(Dense(500))
# model.add(Dense(250))
# model.add(Dense(125))
# model.add(Dropout(0.2))
# model.add(Dense(60))
# model.add(Dense(30))
# model.add(Dense(10))
# model.add(Dense(1))

input1 = Input(shape = (9,))
dense1 = Dense(1000)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(500)(drop1)
dense3 = Dense(250)(dense2)
dense4 = Dense(125)(dense3)
drop2 = Dropout(0.2)(dense4)
dense5 = Dense(60)(drop2)
dense6 = Dense(30)(dense5)
dense7 = Dense(10)(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs = input1, outputs = output1)


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
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1,
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


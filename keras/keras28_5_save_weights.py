# 26-1 copy
# 28-2 copy
# 28-3 copy

from tensorflow.keras.models import Sequential, load_model # 모델을 불러오는 라이브러리
from keras.layers import Dense
import sklearn as sk
print(sk.__version__)  #0.24.2
import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


#1. 데이터 (정규화 과정을 포함)
dataset = load_boston() 
# print(dataset.shape)
print(dataset.DESCR)  # sklearn에서 .describe()와 동일한 데이터의 평균 등을 설명하는 함수
print(dataset.feature_names)  #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
                              #   'B' 'LSTAT']

x = dataset.data
y = dataset.target  #x와 y를 데이터 상에서 분리
# print(x.shape) #(506, 13)
# print(y.shape) #(506,)

x_train, x_test, y_train, y_test =train_test_split(x, y, test_size = 0.2,
                             shuffle=True, random_state=6265)  

#####정규화(07/25)#####
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test) 


#2. 모델
model = Sequential()
# model.add(Dense(100, input_dim = 13))
model.add(Dense(100, input_shape = (13,))) # 이미지의 input_shape = (8,8,1)
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# model.summary()
model.save_weights(".//_save//keras28//keras28_5_save_weights1.h5") # 훈련되지 않은 가중치



#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=16,
          validation_split=0.2, verbose=1)
# hist라고 지정하면서 모델의 훈련에 대한 과정을 입력
end_time = time.time()
model.save(".//_save//keras28//keras28_3_save_model.h5") 
#3. 이후에 모델을 저장하면 가중치까지 저장되어 가장 효과적인 가중치를 저장할 수 있다.
 
model.save_weights(".//_save//keras28//keras28_5_save_weights2.h5") # 훈련된 가중치

#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict([x])


r2 = r2_score(y_test, y_predict)

print("훈련시간", round(end_time-start_time, 2), "초")
print("오차값", loss)
print("결정계수", r2)


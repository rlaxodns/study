# 26-1 copy
# 28-2 copy

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




model = load_model(".//_save//keras28//keras28_3_save_model.h5") 
#3. 컴파일 훈련) 이후에 저장된 내용을 불러오면 가중치까지도 불러올 수 있다.
# 전이학습) 위에서 다른 데이터를 통해서 학습시킨 내용을 가지고 전혀 다른 내용의 
# 데이터를 학습시켜 고도화하는 방법



#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict([x])


r2 = r2_score(y_test, y_predict)

# print("훈련시간", round(end_time-start_time, 2), "초")
print("오차값", loss)
print("결정계수", r2)


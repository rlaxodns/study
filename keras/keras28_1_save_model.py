# 26-1 copy

"""
데이터 전처리
필수적으로 활용할 수 있는 방안, 
컬럼을 확인할 때, 데이터가 한 쪽으로 편중되어 있는 경우, 
또한, 데이터의 크기가 너무 커진다면,
가중치 연산에 크기가 너무 켜져서 문제가 발생할 수 있다.

이러한 문제를 해결하기 위해
1. 표준화: StandardScaler
한 쪽으로 치우쳐진 데이터의 평균을 중심으로 차원을 이동하는 방법,
2. 스케일링 기법:
데이터가 너무 큰 경우 일정한 동일 비율로 나누어 데이터의 크기를 줄이는 방법
cf) 이러한 경우 결과값을 일정한 비율로 나누어 버리면 예측값 자체가 틀어질 수 있어
독립변수들만 일정한 동일비율로 나누어 준다.

cf) 넘파이와 딥러닝은 부동소수점 연산에 특화되어 있다. 이러한 특성을 바탕으로
숫자가 큰 데이터를 연산하는 것보다 소수점 이하의 연산을 유도한다.

MinMaxScaler(정규화: normalization) :
MinMaxScaler = (Xn - X(min))/(X(max) - X(min))
MinMaxScaler는 무조건 최소값을 0으로 잡고 최대값을 1로 잡는다.
따라서 x의 데이터를 0에서 1사이의 데이터로 변환해주는 것(최대 값으로 나누어서),
훈련의 적합한 데이터로 만들어주는 것

<구현방법>
# from keras.preprocessing import MinMaxScaler
# mms = MinMaxScaler()
# x = mms.fit_transform(x)

y 데이터를 건드리는 일은 거의 없음. cf) OneHotEncoding의 경우에는 필수적
"""

#18-1 카피

# 과적합의 해결책 보다는 과적합의 구간을 찾아보는 과정
# loss값을 줄이다가 훈련이 진행이 되지 않는 과정
# 즉, 성능개선이 없는 구간을 과적합 구간이라고 한다.

# 보스턴의 데이터를 다운받아서 보스턴의 집값 예측

from tensorflow.keras.models import Sequential
from keras.layers import Dense
import sklearn as sk
print(sk.__version__)  #0.24.2
import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

######07/25 _ input_shape_
# 우리가 지금까지 인풋을 입력하면서 input_dim을 활용해왔는데, 입력하는 차원은 (열, 컬럼, 특성, 피쳐)
# input_dim의 경우에는 2차원의 데이터만 입력할 수 밖에 없다 
# 따라서 이미지의 경우에는 2차원으로 정렬하면 데이터의 양이 기하급수적으로 늘어나는 문제가 발생한다
# ex) CNN의 경우에는 다차원으로 입력되어 연산을 한기 때문에 input_dim으로 입력하면
# 데이터를 2차원으로 정렬해서 입력하는 복잡함이 발생한다
# 이를 위해 'input_shape'를 활용
######

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

# 훈련데이터 스케일링 이후,
# 범위가 넘어가도록 테스트데이터를 스케일링 하여 과적합 문제를 해결
####################### 

#정규화는 데이터를 분리 이후 훈련데이터에 진행한다(과적합 방지)
# 스케일링
#1. 훈련데이터만 스케일링을 한다.
#2. 이후 .fit_transfrom(x_train)을 진행
#3. 이때, x_test의 비율만 x_train과 동일하게 맞춰준다 // .transform(x_test)

# print(np.min(x_train), np.max(x_train)) #0.0 / 1.0
# print(np.min(x_test), np.max(x_test)) #-0.0019120458891013214 / 1.1479322699591572
# 스케일링은 각각의 컬럼별로 적용되는 것
# Standard는 데이터가 한 쪽으로 쏠린 경우에 더 좋을 수 있지만, 
# 아닌 경우도 있기에 둘 다 사용해보는 것이 좋다

#2. 모델
model = Sequential()
# model.add(Dense(100, input_dim = 13))
model.add(Dense(100, input_shape = (13,))) # 이미지의 input_shape = (8,8,1)
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.summary()

model.save(".//_save//keras28//keras28_1_save_model.h5")


# #3. 컴파일 및 훈련
# model.compile(loss = 'mse', optimizer='adam')
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=500, batch_size=16,
#           validation_split=0.2, verbose=1)
# # hist라고 지정하면서 모델의 훈련에 대한 과정을 입력


# end_time = time.time()

# #4. 예측 및 평가
# loss = model.evaluate(x_test, y_test)
# y_predict = model.predict(x_test)
# result = model.predict([x])


# r2 = r2_score(y_test, y_predict)

# print("훈련시간", round(end_time-start_time, 2), "초")
# print("오차값", loss)
# print("결정계수", r2)

# """<정규화 전>
# 훈련시간 9.87 초
# 오차값 31.81736946105957
# 결정계수 0.6289244398474927

# <정규화 후>
# 훈련시간 9.92 초
# 오차값 24.67317008972168
# 결정계수 0.7122449369631706

# <분리 후 진행>
# 훈련시간 10.7 초
# 오차값 22.435144424438477
# 결정계수 0.6681070297995211

# 훈련시간 11.0 초
# 오차값 22.399995803833008
# 결정계수 0.6686269577326496

# 훈련시간 11.37 초
# 오차값 23.1861515045166
# 결정계수 0.656996965697054

# 훈련시간 10.7 초
# 오차값 22.325042724609375
# 결정계수 0.6697358247730445
# """
# 26-1 copy

# 훈련하는 중에서 가중치가 갱신되는 내용을 지속적으로 저장할 수 있는 기능
# Early Stopping과 유사한 기능

from tensorflow.keras.models import Sequential, load_model
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
print(dataset.feature_names)  

x = dataset.data
y = dataset.target
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




model = load_model("C:\\ai5\\_save\\keras29\\keras29_mcp1.hdf5")
# from keras.model import load_model의 라이브러리가 필요하다 
# 이에 대한 효과는 모델을 저장한 것과 같은 효과가 있다




#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict([x])
r2 = r2_score(y_test, y_predict)

print("오차값", loss)
print("결정계수", r2)

# 오차값 22.29594612121582
# 결정계수 0.6701661973511581
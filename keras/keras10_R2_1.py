# 09_2 카피
# R2(결정계수) 검색
"""
sst: 총변동, 실제 값과 평균과의 오차의 제곱을 모두 합한 값
sse: 설명된 변동, 예측값과 평균값의 오차의 제곱을 모두 합한 값
ssr: 설명되지 않은 변동, 실제값과 예측값의 오차의 제곱을 모두 합한 값

sse/sst= 1 - ssr/sst
sse/sst가 결정계수
x가 y를 얼마나 잘 설명하냐를 의미한다  0~1 사이의 값
즉, 예측모델이 얼마나 데이터에 잘 부합하는지를 수치적으로 보여주는 지표

회귀지표(r2)는 분류지표(accuracy)에서 사용할 수 없음
왜냐하면, 분류는 맞냐 틀리냐의 문제지만
회귀는 '근사치'의 문제이다
"""
import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense



# data
x = np.array([1,2,3,4,6,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=300)

# 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 예측 및 평가
loss = model.evaluate(x_test, y_test)
print('loss', loss)
result = model.predict([x])

y_predict = model.predict([x_test])
from sklearn.metrics import r2_score     #결정계수 구하는 함수
r2 = r2_score(y_test, y_predict)
#보조지표이기 때문에 검증하는 y_test값으로 r2값을 구한다
#또한, 평가는 평가지표로 검증하는 것이 좋기 때문에 
# y_test를 통해서 r2지표를 찾는 것이 좋다.
print('결과', result)
print(r2)

#loss값이 좋지 않은 경우, 1) 데이터 전처리를 하거나 2) 하이퍼 파라미터 튜닝을 한다.
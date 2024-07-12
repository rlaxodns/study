import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. data
x = np.array(range(10))

print(x) #[0 1 2 3 4 5 6 7 8 9]
print(x.shape) #(10,)
# 전산의 모든 것은 0부터 시작하기 때문에 마지막 숫자에서 -1을 해서 끝자리를 판단한다.

x = np.array(range(1,11))
print(x)  #[ 1  2  3  4  5  6  7  8  9 10]
print(x.shape) #(10,)

x = np.array([range(10), range(21,31), range(201,211)]) # 대괄호로 묶지 않으면 각각의 데이터로 남아있다.
#따라서 대괄호로 묶어주면서 하나의 행렬의 형태로 만들어 주어야 한다.
print(x)
"""
[[  0   1   2   3   4   5   6   7   8   9]
 [ 21  22  23  24  25  26  27  28  29  30]
 [201 202 203 204 205 206 207 208 209 
 """
x = x.T  # 잘못된 데이터를 받았을 경우에 행과 열을 전치시킬때 사용
print(x)
"""
[[  0  21 201]
 [  1  22 202]
 [  2  23 203]
 [  3  24 204]
 [  4  25 205]
 [  5  26 206]
 [  6  27 207]
 [  7  28 208]
 [  8  29 209]
 [  9  30 210]]
 """
print(x.shape) #(10, 3)

y = np.array([1,2,3,4,5,6,7,8,9,10])

# [실습]
#[10, 31, 211]을 예측할 것

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=3))

#3. 컴파일 및 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 예측 및 출력
loss = model.evaluate(x, y)
result = model.predict([[10,31,211]])

print("Loss :", loss)
print("Result :", result)

# 에포100, 배치=1
# Loss : 118.2453842163086
# Result : [[8.027553]]

# 에포1000, 배치=1
# Loss : 8.789995627012104e-05
# Result : [[10.981121]]
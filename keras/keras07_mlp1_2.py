import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])

x1 = np.transpose(x)
print(x)
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]]
print(x1)
# [[ 1  6]
#  [ 2  7]
#  [ 3  8]
#  [ 4  9]
#  [ 5 10]]


# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])  # x의 5행2열의 데이터를 입력하고, y의 (5, )의 데이터를 비교하여 
                                                  # result(x,y)를 통해 결과값이 무엇일지 예측
y = np.array([1,2,3,4,5])                         # x와 y의 행렬과 벡터의 차이 존재    #1개의 column은 한 개의 벡터와 같다


print(x.shape)  #(5,2)
print(y.shape)  #(5,) 
# # x와 y의 형태확인
# # 행 무시, 열 우선(그만큼 열(feater, 특성, column)이 중요)

# # 2 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(10))
model.add(Dense(1))

# # 3 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1, y, epochs=1000, batch_size=1) # 행에 따라서 batch_size 조절

# # 4 평가 예측
loss = model.evaluate(x1, y)
results = model.predict([[6,11]])

print("로스 :", loss)
print("x를 통한 y의 예측값 :", results)

# #[실습]: 소수 2째자리까지 맞추기

# # 에포100, 배치1
# # 로스 : 2.1487718186108395e-05
# # [x, y]의 예측값 : [[5.997533]]

# # 에포1000, 배치1
# # 로스 : 4.554578827802025e-13
# # x의 [6, 11]을 통한 y의 예측값 : [[6.]]
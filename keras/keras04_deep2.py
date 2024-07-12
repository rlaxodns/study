from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x=np.array([1,2,3,4,5,6])
y=np.array([1,2,3,5,4,6])

#[실습] 레이어의 깊이와 노드의 갯수를 이용해서 최소의 LOSS를 만들기
# 에포는 100으로 고정
# 소수 넷째자리까지 맞추면 합격 0.33이하

#2. 모델구성 # 딥러닝으로 구현
model=Sequential()
model.add(Dense(2000,input_dim=1)) #Dense(1(output_node), input_dim=1(input_node))
model.add(Dense(1000))             # 히든레이어의 모양을 어떻게 만드는것은 전혀 상관없지만
model.add(Dense(500))              # in-out은 숫자를 맞출것
model.add(Dense(400))  
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))


# epocs 100
# 오차값은 0.3254151940345764
# 예측값은 [[6.8809524]]
# =================
# epocs 100
# 오차값은 0.32392069697380066
# 예측값은 [[6.823294]]


# =================
# epocs 100
# 오차값은 0.00010343971371185035
# 예측값은 [[5.9904633]]
# =================
# epocs 100
# 오차값은 0.0008735790033824742
# 예측값은 [[5.9950857]]

#3. 컴파일 및 훈련
epochs=100
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=epochs)

#4. 예측
loss=model.evaluate(x,y)
result=model.predict(np.array([7]))

print("=================")
print("epocs", epochs)
print("오차값은", loss)
print("예측값은", result)

# epochs=10 오차값은 0.3990181088447571, 예측값은 [[5.774341]]
# 에포=90 오차값은 0.3879551291465759, 예측값은 [[5.837192]]
# 에포=27 오차값은 0.39377501606941223, 예측값은 [[5.8730845]]
# =================
# epocs 10000
# 오차값은 0.3800000250339508
# 예측값은 [[5.7000003]]
# =================
# epocs 1700
# 오차값은 0.4340324401855469
# 예측값은 [[5.200449]]
# =================
# epocs 1800
# 오차값은 0.3878726363182068
# 예측값은 [[5.849583]]
# =================
# epocs 2250
# 오차값은 0.38041967153549194
# 예측값은 [[5.6657915]]
# ==================
# epocs 2270
# 오차값은 0.41429004073143005
# 예측값은 [[6.0116677]]

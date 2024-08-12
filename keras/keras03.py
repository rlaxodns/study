from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x=np.array([1,2,3,4,5])
y=np.array([1,2,4,3,5])

#2. 모델구성
model=Sequential()
model.add(Dense(1,input_dim=1))

#3. 훈련
epochs=20
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=epochs)

#4. 예측
loss=model.evaluate(x,y)
result=model.predict(np.array([6]))

print("=================")
print("epochs", epochs)
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

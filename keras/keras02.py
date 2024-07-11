from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1 데이터
x=np.array([1,2,3,4,5,6])
y=np.array([1,2,3,4,5,6])

#2. 모델을 구성 #
model=Sequential() # 
model.add(Dense(1, input_dim=1)) #y=ax+b       

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100)

#4. 평가 및 예측
loss=model.evaluate(x,y)  #로스 값을 추가하였음, 로스값: 예측치와 원수치의 차이
print("로스", loss)

result=model.predict(np.array([1,2,3,4,5,6,7]))
print("7의 예측값은?", result)
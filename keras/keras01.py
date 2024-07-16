import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
print(tf.__version__)

#1. 데이터
x=np.array([1,2,3])
y=np.array([1,2,3])

#2. 모델 구성
model=Sequential()   #sequential은 순차적 연산
model.add(Dense(1, input_dim=1))  # 입력값 1, 출력값 1

#3. 컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000)   #fit=훈련, x,y=훈련할 데이터, epochs=훈련 횟수

#4. 예측
result=model.predict(np.array([4]))
print(result)

"""
##깃허브 연동 
git config --global user.name"이름"
git config --global user.email"이메일"

git init

git remote add origin 깃허브url

git status #상태확인

git add . or (파일명)
git commit -m "메시지
git push origin master
"""
#08_1 복사


import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense

# data
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])  
#데이터를 물리적으로 train과 test데이터로 분리

#model
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(1))

# compile
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1,
           verbose=0)
#verbose 의 default값은 1, 다른 숫자들은 진행바가 없어짐, 0은 진행바 미표기

# 훈련에 있어서 데이터를 트래인 데이터를 입력

# predict
print("++++++++++++++++++++++++++") # +위로는 테스트를 하는 로스, + 아래로는 평가하는 데이터의 로스를 파악 가능
# 테스트 데이터를 통한 오류값이 중요
loss = model.evaluate(x_test, y_test) # 데이터 평가함에 있어서 테스트 데이터를 통해 평가
result = model.predict([11])

print("loss\n", loss)
print("result\n", result)

# 테스트 데이터와 훈련데이터의 비율에 따라서도 결과값에 영향을 미치는데
# 테스트 데이터는 실질적으로 전체 데이터에서 버려지는 것이므로 
# 훈련데이터가 테스트에 비해 많은것이 좋다
# 특히, 실무에서는 데이터를 모으는 것이 중요하기 때문!!
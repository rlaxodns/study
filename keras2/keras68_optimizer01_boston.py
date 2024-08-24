from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_boston
import tensorflow as tf
tf.random.set_seed(6265) # seed 고정, 성능에 대한 비교를 하기 위해 활용
np.random.seed(6265)

#1. 데이터
data = load_boston()
x = data.data
y = data.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_trian, y_test = train_test_split(x, y, test_size=0.25, random_state=7245)

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
# print(x.shape) #(506, 13)

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim = 13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일 및 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.001 # 기본값, 기본값이 베스트
learning_rate = 0.0
model.compile(loss = 'mse', optimizer=Adam(learning_rate = learning_rate))
model.fit(x_train, y_trian, 
          epochs=100,
          validation_split=0.2,
          batch_size=32,
          )

#4. 평가 및 예측
from sklearn.metrics import r2_score
loss = model.evaluate(x_test, y_test, verbose=0)
print("로스:{0}".format(loss))

y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('r2:{0}'.format(r2))

# 로스:28.413278579711914
# r2:0.7001037447580287
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#1. 데이터
x_data = np.array([range(100), range(301, 401)]).T  #(100, 2)

y1 = np.array(range(3001, 3101))
y2 = np.array(range(13001, 13101))

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test \
    = train_test_split(x_data, y1, y2, test_size=0.2, random_state=4343)

#2-1 모델 구성
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
output1 = Dense(10, activation='relu')(dense2)

#2-2 모델 분기
middle1 = Dense(10, activation='relu')(output1)
middle1_1 = Dense(10, activation='relu')(middle1)
middle1_out = Dense(1)(middle1_1)

middle2 = Dense(10, activation='relu')(output1)
middle2_1 = Dense(10, activation='relu')(middle2)
middle2_out = Dense(1)(middle2_1)

model = Model(inputs = input1, outputs = [middle1_out, middle2_out])

#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x1_train, [y1_train, y2_train],
           epochs=10,
           validation_split=0.2)

#4. 예측 및 평가
x1_pre = np.array([range(100, 105), range(400, 405)]).T

loss = model.evaluate(x1_test, [y1_test, y2_test])
result1, result2 = model.predict([x1_pre])

print('loss', loss,'\n', 'result1', result1, '\n', "result2",result2)


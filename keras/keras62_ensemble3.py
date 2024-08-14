import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#1. 데이터
x1_data = np.array([range(100), range(301, 401)]).T  #(100, 2)
x2_data = np.array([range(101, 201), range(411, 511), 
                    range(150, 250)]).T              #(100, 3)
x3_data = np.array([range(100), range(301, 401), range(77, 177),
                    range(33, 133)]).T # (100, 4)

y1 = np.array(range(3001, 3101))
y2 = np.array(range(13001, 13101))

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test \
    = train_test_split(x1_data, x2_data, x3_data, y1, y2,
                        test_size=0.2, random_state=4343)

#2-1 모델 구성
input1 = Input(shape=(2,))
dense1 = Dense(10)(input1)
dense2 = Dense(20)(dense1)
output1 = Dense(10)(dense2)

#2-2 
input2 = Input(shape = (3,))
dense11 = Dense(10)(input2)
dense21 = Dense(20)(dense11)
output2 = Dense(10)(dense21)

#2-3
input3 = Input(shape = (4,))
dense12 = Dense(10)(input3)
dense22 = Dense(20)(dense12)
output3 = Dense(10)(dense22)

#2-4 모델 합치기
from keras.layers.merge import concatenate
merge1 = concatenate([output1, output2, output3])
merge2 = Dense(10)(merge1)
merge3 = Dense(20)(merge2)

# 모델의 분기점
middle4 = Dense(1)(merge3)
midele41 = Dense(10)(middle4)
output4 = Dense(1)(midele41)

middle5 = Dense(1)(merge3)
middle51 = Dense(10)(middle5)
output5 = Dense(1)(middle51)

model = Model(inputs = [input1, input2, input3], outputs = [output4, output5])


#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train],
           epochs=10,
           validation_split=0.2)

#4. 예측 및 평가
x1_pre = np.array([range(100, 105), range(400, 405)]).T
x2_pre = np.array([range(201, 206), range(511, 516), range(250,255)]).T
x3_pre = np.array([range(100, 105), range(400, 405),
                  range(177, 182), range(133, 138)]).T

loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
result1, result2 = model.predict([x1_pre, x2_pre, x3_pre])

print('loss', loss,'\n', 'result1', result1, '\n', "result2",result2)
# loss [152213296.0, 1532174.625, 150681120.0] 첫번째 로스는 전체로스이고, y1의 로스, y2의 로스

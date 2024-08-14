import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#1. 데이터
x1_data = np.array([range(100), range(301, 401)]).T  #(100, 2)
# ex) 삼성 종가, 하이닉스 종가

x2_data = np.array([range(101, 201), range(411, 511), 
                    range(150, 250)]).T              #(100, 3)
# ex) 원유, 환율, 금시세
# 앙상블은 각각의 데이터에 각각의 모델을 구성한다.

y = np.array(range(3001, 3101)) # 한강물의 온도


## train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_data, x2_data, y, 
                                                                         test_size=0.2, random_state=4343)
# x2_train, x2_test = train_test_split(x2_data)
# y_train, y_test = train_test_split(y)
print(x1_train.shape, x2_train.shape, y_train.shape) #(80, 2) (80, 3) (80,)

#2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name = "bit1")(input1)
dense2 = Dense(20, activation='relu', name = "bit2")(dense1)
dense3 = Dense(30, activation='relu', name = "bit3")(dense2)
output1 = Dense(30, activation='relu', name = "bit4")(dense3)

# model1 = Model(inputs = input1, outputs = output1)

"""
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 2)]               0

 bit1 (Dense)                (None, 10)                30

 bit2 (Dense)                (None, 20)                220

 bit3 (Dense)                (None, 30)                630

 bit4 (Dense)                (None, 30)                930

=================================================================
Total params: 1,810
Trainable params: 1,810
Non-trainable params: 0
_________________________________________________________________
"""
#2-2 모델
input2 = Input(shape = (3,))
dense11 = Dense(10, activation='relu', name = "bit11")(input2)
dense21 = Dense(20, activation='relu', name = "bit21")(dense11)
dense31 = Dense(30, activation='relu', name = "bit31")(dense21)
output2 = Dense(30, activation='relu', name = "bit41")(dense31)

# model2 = Model(inputs = input2, outputs = output2)
# 사실상 모델을 합치면서 모델이 구성되는 것이기 때문에 모델 구성이 필요없다.
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_2 (InputLayer)        [(None, 3)]               0

 bit11 (Dense)               (None, 10)                40

 bit21 (Dense)               (None, 20)                220

 bit31 (Dense)               (None, 30)                630

 bit41 (Dense)               (None, 30)                930

=================================================================
Total params: 1,820
Trainable params: 1,820
Non-trainable params: 0
_________________________________________________________________
"""

#2-3. 두가지 모델 합치기
from keras.layers.merge import Concatenate, concatenate
merge1 = Concatenate(name = 'mg1')([output1, output2], )
# merge1 = concatenate([output1, output2], name = 'mg1')
merge2 = Dense(7, name = 'mg2')(merge1)
merge3 = Dense(20, name = 'mg3')(merge2)
output3 = Dense(1, name = 'last')(merge3)

model3 = Model(inputs = [input1, input2], outputs = output3)
model3.summary()
"""
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 2)]          0           []

 input_2 (InputLayer)           [(None, 3)]          0           []

 bit1 (Dense)                   (None, 10)           30          ['input_1[0][0]']

 bit11 (Dense)                  (None, 10)           40          ['input_2[0][0]']

 bit2 (Dense)                   (None, 20)           220         ['bit1[0][0]']

 bit21 (Dense)                  (None, 20)           220         ['bit11[0][0]']

 bit3 (Dense)                   (None, 30)           630         ['bit2[0][0]']

 bit31 (Dense)                  (None, 30)           630         ['bit21[0][0]']

 bit4 (Dense)                   (None, 30)           930         ['bit3[0][0]']

 bit41 (Dense)                  (None, 30)           930         ['bit31[0][0]']

 mg1 (Concatenate)              (None, 60)           0           ['bit4[0][0]',
                                                                  'bit41[0][0]']

 mg2 (Dense)                    (None, 7)            427         ['mg1[0][0]']

 mg3 (Dense)                    (None, 20)           160         ['mg2[0][0]']

 last (Dense)                   (None, 1)            21          ['mg3[0][0]']

==================================================================================================
Total params: 4,238
Trainable params: 4,238
Non-trainable params: 0
__________________________________________________________________________________________________
"""

#3. 컴파일 및 훈련
model3.compile(loss = 'mse', optimizer='adam')
model3.fit([x1_train, x2_train], y,
           epochs=10000,
           validation_split=0.2)

#4. 예측 및 평가
x1_pre = np.array([range(100, 105), range(400, 405)]).T
x2_pre = np.array([range(201, 206), range(511, 516), range(250,255)]).T

loss = model3.evaluate([x1_test, x2_test], y_test)
result = model3.predict([x1_pre, x2_pre])

print(loss, result)


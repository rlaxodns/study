import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import time

#2. 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10, 10, 1),
                 strides=1,  # strides를 통해 커널사이즈가 분석하는 픽셀 단위를 조절할 수 있음
                 padding='same')) # default는 valid, 'same'을 통해서 원사이즈를 유지할 수 있음
                            # 28X28X1  ==> 27X27X10으로 변환됨
                            # shape = (batch_size, <rows, columns, channels>)
                            # shape = (batch_size, <heights, widths, channels>)
                                                 # <input_shape부분>

model.add(Conv2D(filters = 20, kernel_size = (3, 3)))
model.add(Conv2D(32, (2,2)))

# Dense의 경우, 다양한 차원이 입력이 되지만,
# 사실상 2차원의 데이터로 변환해서 입력해야 한다.

model.add(Flatten())  # 4차원데이터를 2차원으로 변환해주는 메소드

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, input_dim=(64,)))
                        #shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))

model.summary()

"""
커널사이즈를 통해서 데이터를 분석하면서 데이터의 특징이 부각될 수도 있지만, 
데이터가 소실되는 문제가 발생할 수 있다, 이러한 문제를 해결하기 위해 소실없이 진행하기 위해
padding을 통해서 shape를 유지하면서 데이터를 0(이미지의 최소값)으로 채워서 끝까지 분석하는 방법을 채택한다
"""
"""
6X6의 이미지를 2X2로 중첩되게 분석하는 이유,
대체적으로 이미지의 중심부에 많은 정보를 내포하고 있기 때문에 중첩적인 방식을 활용
중첩되지 않게 분석하는 경우는 다른 방식의 분석 방법
따라서, stride는 한 픽셀씩 움직였다는 의미, stride를 통해 중첩되지 않도록 지정할 수도 있음

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          100

=================================================================
Total params: 100
Trainable params: 100
Non-trainable params: 0
_________________________________________________________________
10*10의 이미지를 커널사이즈(3*3)의 스트라이드 2를 했을 경우, 데이터 손실이 발생한다
"""
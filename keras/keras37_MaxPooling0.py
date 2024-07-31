# 이미지에서의 특징적인 부분을 컨볼루션을 돌린다고 해서 크게 특징적인 부분을 찾기 어려움
# 그렇기 때문에 특성값을 구하기 위해서 전체적인 합성곱을 할 필요가 있는지에 대한 의문이 발생
# 큰 데이터를 너무 조금씩 자르고 여러번 연산하는 비효율성에 대한 문제가 발생

# 그래서 특성이 높은 것만 뽑아내는 방안 = MaxPooling
# 해당 방안을 통해서 자원을 효율적으로 활용할 수 있음

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import time

#2. 모델
model = Sequential()
model.add(Conv2D(10, (3,3), input_shape=(28, 28, 1), # 26,26,10
                 strides=1,
                 padding='same'))
model.add(MaxPooling2D())  # MaxPooling()을 추가한다      #13,13,10                 
model.add(Conv2D(filters = 20, kernel_size = (3, 3)))    #11,11,20
model.add(Conv2D(32, (2,2)))                             #10,10,32

# Dense의 경우, 다양한 차원이 입력이 되지만,
# 사실상 2차원의 데이터로 변환해서 입력해야 한다.

model.add(Flatten())  # 4차원데이터를 2차원으로 변환해주는 메소드

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, input_dim=(64,)))
                        #shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))

model.summary()
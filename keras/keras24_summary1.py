from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

"""_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6
# bias값이 레이어에 영향을 미치기 때문에 총연산에 영향을 미쳐서 
# 연산 수가 증가하게 된다. : Node1 * Node2 + bias = param

 dense_1 (Dense)             (None, 4)                 16

 dense_2 (Dense)             (None, 3)                 15

 dense_3 (Dense)             (None, 1)                 4

=================================================================
Total params: 41  # 총 연산횟수
Trainable params: 41
Non-trainable params: 0  
# 정의학습 중에서는 다른 노드의 가중치를 받아오는 연산되어 학습되지 않음을 의미
# 전이학습에서 위의 3가지 요소를 활용, 가중치를 확인하는데 활용
# 가중치를 받아와 활용하는 경우에 있어서는 Non-trainable로 넘어간다
"""
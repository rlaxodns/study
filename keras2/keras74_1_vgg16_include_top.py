import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)


from tensorflow.keras.applications import VGG16

# model = VGG16()
### 디폴트: weights='imagenet',
       #   include_top=True, # True가 디폴트 
       #   input_shape=(224, 224, 3)

# model.summary()
"""
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0
 predictions (Dense)         (None, 1000)              4097000
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_______________________________________________________________
"""

model = VGG16(weights='imagenet',
              include_top=False, # True가 디폴트 #False적용시 flatten()하단의 Dense layer가 사라짐
              input_shape=(100, 100, 3))
model.summary()

####include_top = False ###
# 1. Fc layer 사라짐
# 2. input_shape()를 원하는 크기로 변경 가능
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

(x_train, y_train), (x_test, y_test) =mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) # 흑백의 경우는 1을 생략하는 경우가 있다. == (6000, 28, 28, 1)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts = True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#   dtype=int64))
print(pd.value_counts(y_test))


import matplotlib.pyplot as plt
plt.imshow(x_train[9], 'gray')
plt.show()
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

(x_train,_), (x_test,_) = mnist.load_data()
# print(x_train.shape) #(60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis = 0)

## 스케일링
x = x/255.
print(np.min(x), np.max(x)) # 0.0 1.0

## PCA
x = x.reshape(x.shape[0], 28*28)

pca = PCA(n_components=28*28)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

# argmax max값의 위치를 찾아주는 함수

print(np.argmax(evr_cumsum>= 1.0)+1)   # 713 # 인덱스라서 +1을 해줘야함
print(np.argmax(evr_cumsum>= 0.999)+1) # 486
print(np.argmax(evr_cumsum>= 0.99)+1)  # 331
print(np.argmax(evr_cumsum>= 0.95)+1)  # 154
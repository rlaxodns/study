from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

(x_train,_), (x_test,_) = mnist.load_data()
# print(x_train.shape) #(60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis = 0)
# print(x.shape) (70000, 28, 28)

# x = x.reshape(x.shape[0], 28*28)


# pca = PCA(n_components=28*28)

# x_pca = pca.fit_transform(x)

# evr = pca.explained_variance_ratio_
# evr_cumsum = np.cumsum(evr)
 
# arg1 = np.argmax(evr_cumsum>=0.95)
# arg2 = np.argmax(evr_cumsum>=0.99)
# arg3 = np.argmax(evr_cumsum>=0.999)
# arg4 = np.argmax(evr_cumsum>=1.0)

# print("evr",evr)
# print("evr_cumsum",evr_cumsum)
# print(arg1)
# print(arg2)
# print(arg3)
# print(arg4)

###################################################################
# 힌트 = np.argmax
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
# 이미지에서는 필요없는 부분이 많기 때문에 PCA를 통해서 연산속도와 연산비용을 충분히 줄일 수 있음
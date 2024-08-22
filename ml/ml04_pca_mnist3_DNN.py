from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape) #(60000, 28, 28) (10000, 28, 28)

print(y_train.shape)

## 스케일링
x_train = x_train/255.
x_test = x_test/255.
# print(np.min(x), np.max(x)) # 0.0 1.0

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

results = []
for i in range(5):
    x = [784, 713, 486, 331, 154]

## PCA
    pca = PCA(n_components= x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

# evr = pca.explained_variance_ratio_
# evr_cumsum = np.cumsum(evr)
# print(evr_cumsum)

# # argmax max값의 위치를 찾아주는 함수

# print(np.argmax(evr_cumsum>= 1.0)+1)   # 713 # 인덱스라서 +1을 해줘야함
# print(np.argmax(evr_cumsum>= 0.999)+1) # 486
# print(np.argmax(evr_cumsum>= 0.99)+1)  # 331
# print(np.argmax(evr_cumsum>= 0.95)+1)  # 154

#2. 모델 구성
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(64, input_shape=(x_train1.shape[1],)))
    model.add(Dense(32))
    model.add(Dense(8))
    model.add(Dense(10, activation='softmax'))

#3. 컴파일 및 훈련
    model.compile(loss = 'categorical_crossentropy', optimizer='adam')
    model.fit(x_train1, y_train,
          epochs=1,
          batch_size=16,
          validation_split=0.2)

#4. 평가 및 예측
    loss = model.evaluate(x_test1, y_test)
    # result = model.predict([x_test])
    y_pre = np.argmax(model.predict(x_test1), axis=1).reshape(-1,1)
    y_test1 = np.argmax(y_test, axis=1).reshape(-1,1)

## acc
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test1, y_pre)

      ## 결과를 리스트에 저장
    results.append((i + 1, x[i], acc))

## 전체 결과 출력
for result in results:
    print(f"Iteration: {result[0]}, Components: {result[1]}, Accuracy: {result[2]:.4f}")
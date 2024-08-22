
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

# 모델
data = load_wine()
x = data.data #(178, 13)
y = data.target #(178, 3)

print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48

y = pd.get_dummies(y)
print(y)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=4343,  stratify=y)

####스케일링 적용######
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.fit_transform(x_test)

for i in range(3):
    x = [13, 12, 9]

    #pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

# cum = np.cumsum(pca.explained_variance_ratio_)

# print(np.argmax(cum)+1)  # 13 # 인덱스라서 +1을 해줘야함
# print(np.argmax(cum>= 0.999)+1) # 13
# print(np.argmax(cum>= 0.99)+1)  # 12
# print(np.argmax(cum>= 0.95)+1)  # 9


    #모델 구성
    model = Sequential()
    model.add(Dense(64, input_dim = x_train1.shape[1]))
    model.add(Dense(32))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(16))
    model.add(Dense(3, activation='softmax'))

    #컴파일 및 훈련
    es = EarlyStopping(
        monitor='val_loss',
        mode = 'min',
        patience=100,
        restore_best_weights=True
    )
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['acc'])
    model.fit(x_train1, y_train, epochs=100, batch_size=1,
              validation_split=0.2, callbacks=[es], verbose=0)

    #평가 및 예측
    loss = model.evaluate(x_test1, y_test)
    # result = np.round(model.predict([x]))
    y_pre = np.round(model.predict([x_test1]))
    acc = accuracy_score(y_test, y_pre)

    print("로스", loss, "정확도", acc)

"""
2/2 [==============================] - 0s 2ms/step - loss: 2.0530e-06 - acc: 1.0000
로스 [2.0529869289021008e-06, 1.0] 정확도 1.0
2/2 [==============================] - 0s 2ms/step - loss: 1.7020e-06 - acc: 1.0000
로스 [1.7019942788465414e-06, 1.0] 정확도 1.0
2/2 [==============================] - 0s 996us/step - loss: 3.2280e-04 - acc: 1.0000
로스 [0.0003228009445592761, 1.0] 정확도 1.0
"""

# # 로스 [0.0024288140702992678, 1.0] 정확도 1.0
# # 로스 [2.1258176730043488e-06, 1.0] 정확도 1.0
# # 로스 [0.10237397253513336, 0.9722222089767456] 정확도 0.9722222222222222
# # 로스 [1.4768315850233193e-06, 1.0] 정확도 1.0
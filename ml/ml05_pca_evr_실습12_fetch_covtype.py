from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder

# 데이터 구성
data = fetch_covtype()
x = data.data
y = data.target
# print(x.shape) #(581012, 54)
# print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

y = pd.get_dummies(y)



# ohe = OneHotEncoder(sparse=False)
# y=y.reshape(-1,1)
# y = ohe.fit_transform(y)
# print(y.shape)
# print(y)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y.shape)
# print(y)
# print(type(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=4345, stratify=y) # 분류에서만 쓰는 파라미터, 
# y라벨의 데이터를 정확하게 비율대로 나누어 준다.

print(x_train.shape, y_train.shape) #(464809, 54) (464809,)
print(x_test.shape, y_test.shape) #(116203, 54) (116203,)
# print(pd.value_counts(y_train))
# 2    226647
# 1    169475
# 3     28619
# 7     16414
# 6     13853
# 5      7622
# 4      2179

###스케일링 적용###
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)

for i in range(4):
    x = [52, 43, 28, 17]
    #pca
    from sklearn.decomposition import PCA
    pca = PCA()
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    # cum = np.cumsum(pca.explained_variance_ratio_)

    # print(np.argmax(cum>=1)+1)  # 52 # 인덱스라서 +1을 해줘야함
    # print(np.argmax(cum>= 0.999)+1) # 43
    # print(np.argmax(cum>= 0.99)+1)  # 28
    # print(np.argmax(cum>= 0.95)+1)  # 7

    # 모델 구성
    model = Sequential()
    model.add(Dense(64, input_dim = x_train1.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    # 컴파일 및 훈련
    es = EarlyStopping(
        monitor='val_loss',
        mode = 'min', 
        patience=100 , 
        restore_best_weights= True
    )
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
    model.fit(x_train1, y_train, epochs=1, batch_size=1024,
              validation_split=0.2, callbacks=[es], verbose=0)

    # 평가 및 예측
    loss = model.evaluate(x_test1, y_test)
    y_pre = np.round(model.predict(x_test1))
    acc = accuracy_score(y_test, y_pre)
    print(loss, acc)

"""
[0.540134072303772, 0.7723294496536255] 0.7576568591172345
3632/3632 [==============================] - 8s 2ms/step - loss: 0.5515 - acc: 0.7685
[0.5514649748802185, 0.7684913277626038] 0.7594640413758681
3632/3632 [==============================] - 8s 2ms/step - loss: 0.5504 - acc: 0.7683
[0.5503753423690796, 0.7683278322219849] 0.7467449205270088
3632/3632 [==============================] - 7s 2ms/step - loss: 0.5461 - acc: 0.7705
[0.5461302995681763, 0.770513653755188] 0.7545158042391332
"""

# # [0.13793563842773438, 0.9503799676895142] 0.9499754739550614
# # [0.23125745356082916, 0.9136000275611877] 0.9136
# # [0.1510281264781952, 0.9443387985229492] 0.9436933641988589
# # [0.13430580496788025, 0.9536586999893188] 0.9533058526888291
# # 
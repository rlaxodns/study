from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
import random as rn
rn.seed(6265)
tf.random.set_seed(6265)
np.random.seed(6265)
from tensorflow.keras.optimizers import Adam
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

# 데이터 구성
data = fetch_covtype()
x = data.data
y = data.target

y = pd.get_dummies(y)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=4345, stratify=y) # 분류에서만 쓰는 파라미터, 
# y라벨의 데이터를 정확하게 비율대로 나누어 준다.



###스케일링 적용###
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)



# 모델 구성
model = Sequential()
model.add(Dense(64, input_dim = 54, activation='relu'))
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
for i in range(0, len(lr), 1):
    
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=lr[i]), metrics=['acc'])
    model.fit(x_train, y_train, epochs=1, batch_size=1024, verbose=0,
            validation_split=0.2, callbacks=[es])

    # 평가 및 예측
    print("===========출력==================")
    loss1 = model.evaluate(x_test, y_test, verbose=0)
    print('lr:{0},로스:{1}'.format(lr, loss1[0]))
    print('lr:{0},r2:{1}'.format(lr, loss1[1]))
    # [0.13793563842773438, 0.9503799676895142] 0.9499754739550614
    # [0.23125745356082916, 0.9136000275611877] 0.9136
    # [0.1510281264781952, 0.9443387985229492] 0.9436933641988589
    # [0.13430580496788025, 0.9536586999893188] 0.9533058526888291
    # 

    """
    ===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:1.2107232809066772
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.4876036047935486
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:1.2053619623184204
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.4876036047935486
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:1.20587956905365
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.4876036047935486
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:1.2051756381988525
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.4876036047935486
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:1.2051618099212646
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.4876036047935486
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:1.2051573991775513
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.4876036047935486"""
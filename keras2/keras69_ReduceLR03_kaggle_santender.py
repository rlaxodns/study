# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time

import tensorflow as tf
import random as rn
rn.seed(6265)
tf.random.set_seed(6265)
np.random.seed(6265)

# 데이터 구성
path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"
train = pd.read_csv(path+"train.csv", index_col=0)
test = pd.read_csv(path+"test.csv", index_col=0)
sub = pd.read_csv(path+"sample_submission.csv", index_col = 0)

x = train.drop(['target'], axis = 1)
y = train['target']
print(pd.value_counts(y))
print(x.shape,y.shape) #(200000, 200) (200000,)

# y = pd.get_dummies(y)
# print(y.shape) #(200000, 2)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                                    random_state=4343)

####스케일링 적용####
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)
test = rbs.transform(test)



# 모델
model = Sequential()
model.add(Dense(64, input_dim = 200))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))

# 컴파일 및 훈련
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
lr = 0.1

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=50,
    restore_best_weights=True
)
rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode = 'auto',
    patience=30,
    verbose=1,
    factor = 0.8
)
model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=lr), metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=64, 
            validation_split= 0.2, callbacks=[es,rlr], verbose=0)

    # 평가 및 예측
print("===========출력==================")
loss1 = model.evaluate(x_test, y_test, verbose=0)
print('lr:{0},로스:{1}'.format(lr, loss1[0]))
print('lr:{0},r2:{1}'.format(lr, loss1[1]))
# sub['target'] = result
# sub.to_csv(path +"sample_submission03.csv")

# [0.23117949068546295, 0.9138249754905701] 0.913825
# [0.23127810657024384, 0.9140750169754028] 0.914075
# [0.23467880487442017, 0.9129999876022339] 0.913
# [0.23132435977458954, 0.9133999943733215] 0.9134
"""
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:4.447984218597412
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.8654500246047974
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:0.2531857192516327
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.9070500135421753
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:0.23351562023162842
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.9127500057220459
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:0.23168203234672546
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.9136250019073486
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:0.23115813732147217
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.9136999845504761
===========출력==================
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],로스:0.2312607616186142
lr:[0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001],r2:0.9137499928474426

"""
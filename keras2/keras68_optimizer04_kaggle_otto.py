import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import random as rn
rn.seed(6265)
tf.random.set_seed(6265)
np.random.seed(6265)

# 데이터 구성
path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sub = pd.read_csv(path + "sampleSubmission.csv", index_col=0, )

le = LabelEncoder()
train["target"] = le.fit_transform(train["target"])

x = train.drop(['target'], axis=1)
y = train['target']


y = pd.get_dummies(y)
print(x.shape, y.shape) #(61878, 93) (61878, 9)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=7777, stratify=y)

####스케일링 적용####
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)
test = rbs.transform(test)


#. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim = 93, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(9, activation='softmax'))

# 컴파일 및 훈련
from tensorflow.keras.optimizers import Adam
lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=32,
    restore_best_weights=True
)

for i in range(0, len(lr), 1):

    model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=lr[i]),
                metrics=['acc'])
    model.fit(
        x_train,
        y_train,
        callbacks=[es],
        epochs=100,
        batch_size=256,    
        validation_split=0.3,
        verbose=0)

    # 평가 및 예측
    print("===========출력==================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr:{0},로스:{1}'.format(lr[i], loss[0]))
    print('lr:{0},r2:{1}'.format(lr[i], loss[1]))

# sub[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8",
#      "Class_9"]] = result
# sub.to_csv(path + "sub0011.csv")

# 0.5722733736038208 0.793228805065155
# 0.6055837869644165 0.7788461446762085
# 0.5790331959724426 0.7954912781715393
# 0.5654720664024353 0.7845022678375244

"""
===========출력==================
lr:0.1,로스:1.9525494575500488
lr:0.1,r2:0.26050421595573425
===========출력==================
lr:0.01,로스:1.9505445957183838
lr:0.01,r2:0.26050421595573425
===========출력==================
lr:0.005,로스:1.9505760669708252
lr:0.005,r2:0.26050421595573425
===========출력==================
lr:0.001,로스:1.9504979848861694
lr:0.001,r2:0.26050421595573425
===========출력==================
lr:0.0005,로스:1.95050048828125
lr:0.0005,r2:0.26050421595573425
===========출력==================
lr:0.0001,로스:1.9505023956298828
lr:0.0001,r2:0.26050421595573425
"""
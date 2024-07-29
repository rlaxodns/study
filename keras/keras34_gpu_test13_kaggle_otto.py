import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time as t
# 데이터 구성
stt = t.time()
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
from keras.layers import Dropout, Input
from keras.models import Model

# model = Sequential()
# model.add(Dense(128, input_dim = 93, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(9, activation='softmax'))


input1 = Input(shape=(93,))
dense1 = Dense(218)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 =Dense(64)(drop1)
dense3 = Dense(64)(dense2)
dense4 = Dense(64)(dense3)
dense5 = Dense(64)(dense4)
dense6 = Dense(64)(dense5)
drop2 = Dropout(0.2)(dense6)
dense7 = Dense(64)(drop2)
dense8 = Dense(64)(dense7)
dense9 = Dense(64)(dense8)
output1 = Dense(9, activation='softmax')(dense9)
model = Model(inputs = input1, outputs = output1)

# 컴파일 및 훈련
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode = 'auto', 
    patience = 20, 
    verbose = 1, 
    save_best_only=True,
    filepath=".//_save//keras32//keras32_dropout13_save_kaggle_otto.hdf5"
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=32,
    restore_best_weights=True
)
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',
              metrics=['acc'])
model.fit(
    x_train,
    y_train,
    callbacks=[es,mcp],
    epochs=100,
    batch_size=256,    
    validation_split=0.3,
    )

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(test)
y_pre = model.predict(x_test)
et = t.time()

print(loss[0], loss[1])
print(et-stt)
sub[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8",
     "Class_9"]] = result
sub.to_csv(path + "sub0011.csv")
# <cpu> 19.630926609039307
# 0.5722733736038208 0.793228805065155
# 0.6055837869644165 0.7788461446762085
# 0.5790331959724426 0.7954912781715393
# 0.5654720664024353 0.7845022678375244
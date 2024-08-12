from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
#데이터
x, y = load_digits(return_X_y=True)
print(x.shape, y.shape) #(1797, 64) (1797,)
print(pd.value_counts(y, sort=False))

ohe = OneHotEncoder(sparse=False)
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)
print(x.shape, y.shape) #(1797, 64) (1797, 10)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=6666, 
                                                    stratify=y)

###스케일링 적용###
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)

x_train = x_train.reshape(
    x_train.shape[0], 
    x_train.shape[1],
    1
)
x_test = x_test.reshape(
    x_test.shape[0], 
    x_test.shape[1],
    1
)

print(x_train.shape, x_test.shape)


# 모델 구성
from keras.layers import Dropout
from keras.layers import LSTM
model = Sequential()

model.add(LSTM(32, input_shape = (x_train.shape[1], 1)))

model.add(Dense(128, input_dim = 64))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(10, activation='softmax'))

# 컴파일 및 훈련
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode = 'auto', 
    patience = 50, 
    verbose=1,
    save_best_only= True, 
    filepath=".//_save//keras32//keras32_dropout10_save_digits.hdf5"
)

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=100,
    restore_best_weights=True
)
model.compile(loss = 'categorical_crossentropy', optimizer='adam'
              , metrics=['acc'])
model.fit(x_train, y_train,
        epochs=1000,
        batch_size=1,
        validation_split=0.2,
        callbacks=[es, mcp])

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pre)

print(loss, acc)

# [2.80710506439209, 0.9527778029441833] 0.9527777777777777
# [104.78390502929688, 0.5361111164093018] 0.5361111111111111
# [3.353447198867798, 0.9416666626930237] 0.9416666666666667
# [24.242084503173828, 0.9222221970558167] 0.9222222222222223
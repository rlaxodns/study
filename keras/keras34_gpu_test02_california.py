from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time as t
st_t = t.time()

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,
                                    random_state=6235)

##스케일링 적용##
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)

# [실습]r2 059이상
#모델
# input_dim = 8, Dense(1))
from keras.layers import Dropout, Input
from keras.models import Model

input1 = Input(shape= (8,))
dense1 = Dense(100)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(100)(drop1)
drop2 = Dropout(0.1)(dense2)
dense3 = Dense(100)(drop2)
dense4 = Dense(100)(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs = input1, outputs = output1)

model.summary()

# 컴파일 및 훈련
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor= 'val_loss',
    mode = min,
    patience=3,
    restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor= 'val_loss',
    mode = 'auto',
    patience = 10,
    verbose= 1,
    save_best_only=True,
    filepath = ".//_save//keras32//keras32_dropout02_save_california.hdf5"
)

model.compile(loss='mse', optimizer='adam')
st_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=10, 
          validation_split=0.2, verbose=0, callbacks=[es, mcp])
end_time = time.time()

# 예측 및 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict([x_test])

from sklearn.metrics import r2_score
e_t = t.time()
r2 = r2_score(y_test, y_predict)

print(loss)
print(r2)
print(e_t-st_t)
#<cpu> 6.127507925033569 <gpu>17.8498592376709
"""0.5996702909469604
0.561659386010252

0.6130666732788086
0.5518669095763122

0.6405205726623535
0.5317990030598638

<스케일링 적용 후>
0.5360893607139587
0.5916927408072075

0.5275655388832092
0.5981848140146168
4.71520209312439
"""


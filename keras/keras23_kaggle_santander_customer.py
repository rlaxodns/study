# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time

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
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=50,
    restore_best_weights=True
)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, batch_size=64, 
          validation_split= 0.2, callbacks=[es])

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
result = np.round(model.predict(test))
y_pre = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pre)

print(loss, acc)

sub['target'] = result
sub.to_csv(path +"sample_submission03.csv")
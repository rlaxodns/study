# https://dacon.io/competitions/official/236068/overview/description

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

#1. 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\train.csv", index_col = 0)
test = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\test.csv", index_col=0)
submission = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\sample_submission.csv", index_col = 0)

print(train.info(), train.shape) #None (652, 9)
print(train.columns)
"""Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')"""

x = train.drop(['Outcome'], axis=1)
y = train['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, 
                                                   shuffle=True, random_state=1512)

#2. 모델 구성
model =Sequential()
model.add(Dense(64, input_dim = 8, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 및 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode ='min',
    patience= 110,
    restore_best_weights=True
)
model.compile(loss = 'mse', optimizer = 'adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 1000, batch_size=1,
           validation_split=0.25, callbacks=[es])

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict([x_test]))

result = np.round(model.predict([test]))

acc = accuracy_score(y_test, y_pre)
print(y_pre)
print(loss)
print(acc)

submission['Outcome'] = result
submission.to_csv("C:\\ai5\\_data\\dacon\\diabetes\\sample_submission1555.csv")


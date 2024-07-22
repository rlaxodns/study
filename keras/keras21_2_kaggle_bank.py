# train['종가'] = train['종가'].str.replace(',', '')
# https://www.kaggle.com/c/playground-series-s4e1/data
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping

# 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\train.csv", index_col=[0, 1, 2])
test = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\test.csv", index_col= [0, 1, 2])
submission = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\sample_submission.csv", index_col=[0])

print(train.dtypes)
"""Surname             object
CreditScore          int64
Geography           object
Gender              object
Age                float64
Tenure               int64
Balance            float64
NumOfProducts        int64
HasCrCard          float64
IsActiveMember     float64
EstimatedSalary    float64
Exited               int64"""
# France = 1, Spain = 2, Germany = 3 // Male = 1, Female =2

train['Geography'] = train['Geography'].str.replace('France', '1')
train['Geography'] = train['Geography'].str.replace('Spain', '2')
train['Geography'] = train['Geography'].str.replace('Germany', '3')

train['Gender'] = train['Gender'].str.replace('Male', '1')
train['Gender'] = train['Gender'].str.replace('Female', '2')

train = train.astype(float)
print(train.dtypes)

test['Geography'] = test['Geography'].str.replace('France', '1')
test['Geography'] = test['Geography'].str.replace('Spain', '2')
test['Geography'] = test['Geography'].str.replace('Germany', '3')

test['Gender'] = test['Gender'].str.replace('Male', '1')
test['Gender'] = test['Gender'].str.replace('Female', '2')
test = test.astype(float)
print(test.dtypes)


x = train.drop(['Exited'], axis = 1)
y = train['Exited']

from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler()
x[:] = scalar.fit_transform(x[:])

print(x.shape, y.shape) #(165034, 10) (165034,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
                                                    random_state= 6666)

#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim = 10, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(4000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 및 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=20,
    restore_best_weights=True
)
model.compile(loss = 'mse', optimizer = 'adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 30, batch_size=120,
          validation_split=0.3, callbacks=[es])

#4. 평가 및 훈련
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict([x_test]))
result = np.round(model.predict([test]))

submission['Exited'] = result
submission.to_csv("C:\\ai5\\_data\\kaggle\\bank\\sample_submission_153311.csv")
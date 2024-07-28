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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
                                                   shuffle=True, random_state=4343)
# print(pd.DataFrame(x).value_counts)

###스케일링 적용####
from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)


#2. 모델 구성
from keras.layers import Dropout, Input
from keras.models import Model
# model =Sequential()
# model.add(Dense(64, input_dim = 8, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

input1 = Input(shape = (30,))
dense1 = Dense(1000)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(900)(drop1)
dense3 = Dense(800)(dense2)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(470)(drop2)
dense5 = Dense(260)(dense4)
drop3 = Dropout(0.2)(dense5)
dense6 = Dense(150)(drop3)
dense7 = Dense(50)(dense6)
output1 = Dense(1, activation='sigmoid')(dense7)
model = Model(inputs = input1, outputs = output1)


#3. 컴파일 및 훈련
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor = 'val_loss', 
    mode = 'auto', 
    patience = 100, 
    verbose=1, 
    save_best_only=True, 
    filepath = ".//_save//keras32//keras32_dropout07_save_dacon_diabets.hdf5"
)

es = EarlyStopping(
    monitor='val_loss',
    mode ='min',
    patience= 200,
    restore_best_weights=True
)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 10000, batch_size=1,
           validation_split=0.2, callbacks=[es, mcp])

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

"""
[0.5546854734420776, 0.7099236845970154]
0.7099236641221374

[0.5881093740463257, 0.694656491279602]
0.6946564885496184

[0.5346061587333679, 0.7633587718009949]
0.7633587786259542

<적용 후>
[0.4714379608631134, 0.7786259651184082]
0.7786259541984732

[0.5277946591377258, 0.7633587718009949]
0.7633587786259542
"""
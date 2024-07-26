# 26-1 copy

# 훈련하는 중에서 가중치가 갱신되는 내용을 지속적으로 저장할 수 있는 기능
# Early Stopping과 유사한 기능

from tensorflow.keras.models import Sequential
from keras.layers import Dense
import sklearn as sk
print(sk.__version__)  #0.24.2
import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


#1. 데이터 (정규화 과정을 포함)
dataset = load_boston() 
# print(dataset.shape)
print(dataset.DESCR)  # sklearn에서 .describe()와 동일한 데이터의 평균 등을 설명하는 함수
print(dataset.feature_names)  

x = dataset.data
y = dataset.target
# print(x.shape) #(506, 13)
# print(y.shape) #(506,)

x_train, x_test, y_train, y_test =train_test_split(x, y, test_size = 0.2,
                             shuffle=True, random_state=6265)  

#####정규화(07/25)#####
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test) 


#2. 모델
model = Sequential()
model.add(Dense(100, input_shape = (13,))) # 이미지의 input_shape = (8,8,1)
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.summary()

model.save(".//_save//keras28//keras28_1_save_model.h5")


#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss',
    mode ='min',
    patience=50, 
    verbose = 1, # Restoring model weights from the end of the best epoch: 112
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    patience = 10,
    verbose = 1,
    save_best_only = True,
    filepath=".\\_save\\keras29\\keras29_mcp3.hdf5"
)


hist = model.fit(x_train, y_train,
                epochs=500,
                batch_size=16,
                validation_split=0.2, 
                callbacks=[es, mcp])


model.save(".//_save//keras29//keras29_3_save_model.h5")
#restore_best_weights이기 때문에 가장 최적의 가중치가 저장된다

#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict([x])
r2 = r2_score(y_test, y_predict)

print("오차값", loss)
print("결정계수", r2)

# 오차값 22.29594612121582
# 결정계수 0.6701661973511581
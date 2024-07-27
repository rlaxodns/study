# 우리가 데이터 처리와 모델 구성에 있어서의 문제는 Overfit(과적합)이다.
# 데이터를 훈련시킬 수록, 레이어를 늘릴 수록,
# 데이터를 늘릴 수록 loss 또는 Accuracy가 갱신되지 않는 문제가 발생한다.

# 이를 해결할 수 있는 방안 중 하나, Dropout
# 전체를 훈련시키는 상황에서 랜덤한 노드를 제외하여 훈련시키는 경우에서 
# 더욱 나은 성능을 보이는 경우가 있다
# 과적합에 뛰어난 성능을 보인다 그러나, 이럴거면 처음부터 노드의 수를 줄일 수 있으나, 
# 드랍아웃의 위치를 랜덤하게 바뀌면서 과적합의 문제를 해결할 수 있다.
# 그러나 모델 구성 및 훈련에 있어서만 적용하게되는 것이고 이외의 평가-예측에서는 적용되지 않는다
# 왜냐하면 모델 그 자체를 평가하는 것이지, 드랍아웃을 적용되는 것은 아니다 
# cf) 텐서플로의 경우에서는 자동으로 평가-예측에서는 적용되지 않도록 지정되어 있지만, 
# pytoch의 경우에서는 함수를 적용해야한다



# 29-5 copy

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

x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test) 


#2. 모델
from keras.layers import Dropout

model = Sequential()
model.add(Dense(100, input_shape = (13,), activation='relu')) # 이미지의 input_shape = (8,8,1)
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

model.summary()




#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 훈련하는 중에서 가중치가 갱신되는 내용을 지속적으로 저장할 수 있는 기능
# Early Stopping과 유사한 기능

es = EarlyStopping(
    monitor='val_loss',
    mode ='min',
    patience=50, 
    verbose = 1, # Restoring model weights from the end of the best epoch: 112
    restore_best_weights=True
)



####자동으로 mcp파일 세이브 만들기####
import datetime
date = datetime.datetime.now() 
date = date.strftime("%m%d_%H%M") # 시간을 문자열로 바꾸어주는 함수 0726_1654

path = ".\\_save\\keras32\\"
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    #ex) '1000-0.xxxx.hdf5', 랜덤스테이트도 적용가능
filepath = "".join([ path, "keras32_", date, '_', filename])
# 생성 ex) .\\_save\\keras29\\keras29_1000-0.xxxx.hdf5
# to.csv에도 적용 가능

#####################################

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode = 'auto',
    patience = 10,
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)


model.fit(x_train, y_train,
                epochs=500,
                batch_size=16,
                validation_split=0.2, 
                callbacks=[es, mcp]
                )


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

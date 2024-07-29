# 데이터에 있어서 두개(1, 0)를 나타내는 데이터에서 두개와 다른 데이터가 존재하는 경우에 있어서 
# 결과값에 지대한 영향을 미치게 되고 해당 문제를 찾기 위해 어려움이 발생할 수 있다

# 두개 중에 하나를 찾는 것, '이진분류'
# 두개 이상의 데이터를 분류 하는 것은 '다중분류'
# 지금까지와 달리 최종 결과값을 비교하여 맞는지 틀린지를 검토한다
# accuracy(정확성) 지표를 사용한다

import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import time as t
#1 데이터
st_t = t.time()
dataset = load_breast_cancer()
print(dataset.DESCR) # pandas의 .describe()와 동일
# :Number of Instances(행): 569
# :Number of Attributes(속성): 30 numeric(열), predictive attributes and the class
print(dataset.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

x = dataset.data
y = dataset.target
print(x.shape, y.shape) #(569, 30) (569,)
# 만약 데이터 내에 이상치 혹은 다중분류가 존재하는 경우의 문제는? 

# 특정 요소 갯수 세기 .count  / 이상치 탐지, z-scort, iqr


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 6555)

#0과 1의 갯수가 몇개인지 찾는 방법
print(np.unique(y, return_counts = True)) #(array([0, 1]), array([212, 357], dtype=int64))
# 데이터 불균형의 문제로, 정확성(accuracy)의 확인하는데 있어서 
# 문제를 야기할 수 있기 때문에 '데이터의 라벨과 갯수'를 확인해야한다
# 데이터 불균형의 문제를 '데이터 증폭'기술을 통해서 해결할 수 있다
print(np.unique(x, return_counts = True))
# (array([0.000e+00, 6.920e-04, 7.929e-04, ..., 3.234e+03, 3.432e+03,
#    4.254e+03]), array([78,  2,  1, ...,  1,  1,  1], dtype=int64))

print(type(x)) #<class 'numpy.ndarray'> 
# print(y.value_counts()) #에러
print(pd.DataFrame(y).value_counts()) # 판다스로 확인하기 위해서는 데이터프레임으로 확인
# 1    357
# 0    212
# dtype: int64
print(pd.Series(y).value_counts)
print(pd.value_counts(y)) 
# 1    357
# 0    212
# dtype: int64

# pandas에서는 벡터 형태를 series라 하고, metrics형태를 Dataframe이라 함
# 시리즈는 데이터프레임이 맞지만, 데이터프레임은 시리즈가 아니다 
# 따라서 데이터 프레임의 한개의 컬럼은 시리즈와 동일하다

print(x_train.shape, y_train.shape) #(398, 30) (398,)
print(x_test.shape, y_test.shape) #(171, 30) (171,)

####스케일링 적용#####
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
####################



#2. 모델
from keras.layers import Dropout, Input
from keras.models import Model
# model = Sequential() 
# model.add(Dense(32, input_dim = 30, activation='relu'))
# model.add(Dropout(0.2)) 
# model.add(Dense(32, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2)) 
# model.add(Dense(128, activation='sigmoid'))
# model.add(Dense(64, activation='sigmoid')) # sigmoid 함수를 중간에 사용해도 괜찮다
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2)) 
# model.add(Dense(16,activation='relu'))    # 활성화함수 등을 이용하여 더욱 다양한 히든 레이어를 구축할 수 있다
# model.add(Dense(1, activation='sigmoid')) # activation의 default는 linear

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
# model.compile(loss = 'mse', optimizer='adam', metrics=['accuracy']) #메트릭스에 accuracy를 넣어주면
# 훈련에 영향을 미치지 않음 또한, 0과 1사이에 값이 있을 경우를 반올림하여서 계산하여 판단한다 # acc만 넣어도 정확성 판단

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc']) 


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor= 'val_loss',
    mode = min, # 모르면 auto, 정확도는 자동으로 최대값 // 기본값은 auto
    patience = 100, #참을성이 적으면 좋은 로스값을 얻을 수 없다.
    # 35.403099060058594
    restore_best_weights=True, # 실제적으로 연산은 멈춘 지점에서 가중치가 잡히기 때문에
    #   최소값 지점으로 가중치로 잡아준다. 
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto', 
    patience = 20, 
    verbose=1, 
    save_best_only=True, 
    filepath=".//_save//keras32//keras32_dropout06_save_cancer.hdf5"
)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=32,
          validation_split=0.2, verbose=1, callbacks=[es, mcp])
# hist라고 지정하면서 모델의 훈련에 대한 과정을 입력




#4. 예측 및 평가
loss = model.evaluate(x_test, y_test) #[0.3684210479259491, 0.6315789222717285(accuracy)]

y_predict = model.predict(x_test)
print(y_predict)

y_predict = np.round(y_predict)
# 넘파이 형태의 자료형이기에 np.round를 통해서 반올림
result = model.predict([x])
# r2 = r2_score(y_test, y_predict)
# print(r2)
# print(result)
print("로스:", loss[0])
print("ACC:", loss[1])
e_t = t.time()
from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
print("acc_score", accuracy_score)
print(e_t - st_t)

# print(y_predict)
#ValueError: Classification metrics can't handle a mix of binary and continuous targets
# 즉, 분류데이터에서는 분류값(0과 1)만 입력하라는 것을 의미

# 지금까지의 모형은 선을 그어서 답을 예측하는것이지만, 현재의 분류에서는 0과 1에서 
# 예측하는 것이기 때문에 예측치를 0과 1 사이 값으로 특정 기준치를 통해 변환하는 과정이 필요하다
# 방법은 activation (활성화 함수, 한정함수)
# 이때 한정하는 것은 loss = mse이기 때문에
# 시그모이드 함수를 통해서 0~1사이의 실수에 한정한다

# <cpu>10.943735837936401 <gpu>7.503032684326172
"""
적용전)
로스: 0.06252694129943848
ACC: 0.9122806787490845
acc_score 0.9122807017543859

후)
로스: 0.17258988320827484
ACC: 0.9122806787490845
acc_score 0.9122807017543859

로스: 0.2512463927268982
ACC: 0.9210526347160339
acc_score 0.9210526315789473


<스케일링 적용>
로스: 0.037650082260370255
ACC: 0.9912280440330505
acc_score 0.9912280701754386

로스: 0.1361871361732483
ACC: 0.9473684430122375
acc_score 0.9473684210526315

"""
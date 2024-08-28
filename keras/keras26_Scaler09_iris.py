# 선형회귀한 임의의 선을 긋고서 일치하는 지에 대한 여부 
# 이진분류와 회귀의 차이점은 시그모이드 함수
# 다중분류
"""Binary Cross Entropy loss는 
무조건 이진분류에서 활용
"""

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

dataset = load_iris()
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_names)

x = dataset.data 
y = dataset.target
#############원핫인코딩##################################
#판다스를 활용한 기법
y = pd.get_dummies(y)

# #사이킷런활용
# from sklearn.preprocessing import OneHotEncoder 
# ohe = OneHotEncoder(sparse=False) #True가 기본값
# y = y.reshape(-1, 1)
# y = ohe.fit_transform(y) # 사이킷런에서는 메트릭스형태의 데이터를 받아들인다
#reshape의 조건) 데이터의 내용이 바뀌면 안되며, 데이터의 순서가 바뀌면 안된다.

#케라스를 활용한 기법
# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y)
########################################################
print(y.shape)
print(y)


# print(type(x))
# print(y)
# print(np.unique(y, return_counts = True)) #(array([0, 1, 2]), array([50, 50, 50], dtype=int64))
# print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=4343,  stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = mas.fit_transform(x_train)
x_test = mas.transform(x_test)

print(x_train.shape, y_train.shape)
exit()
# 모델 구성
model = Sequential()
model.add(Dense(128, input_dim = 4, activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(3, activation='softmax'))
          
                
# 컴파일 및 훈련
es = EarlyStopping(
    monitor= 'val_loss',
    mode='min', 
    patience=100,
    restore_best_weights=True
)
model.compile(loss = 'categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2, callbacks=[es])

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
result = np.round(model.predict([x]))
y_pre = np.round(model.predict([x_test]))

acc = accuracy_score(y_test, y_pre)

print(loss[0])
print(loss[1])
print("로스",loss, "정확도", acc)

# 0.038622718304395676
# 1.0
# 로스 [0.038622718304395676, 1.0] 정확도 1.0
# 로스 [0.06536506116390228, 0.9666666388511658] 정확도 0.9666666666666667
# 로스 [0.38589128851890564, 0.8666666746139526] 정확도 0.8666666666666667
# 로스 [0.09475670754909515, 0.9333333373069763] 정확도 0.9333333333333333
# 0.050961993634700775
# 0.9666666388511658
# 로스 [0.050961993634700775, 0.9666666388511658] 정확도 0.9666666666666667
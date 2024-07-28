"""1,2,3가지의 총연계
1. One Hot Encoading 
# - 분류에 대한 수치를 부여하는 것인데 비해서, 연산과정에서의 다른 판단이 내려지는 경우를 막기 위해,
# 각각의 분류에 대한 독립성을 부여하여 범주화하는 것을 의미한다.
# 각각의 위치에 있는 인덱스에 1을 부여하고 나머지 자리에는 0을 대입한다
# ex) 0[1,0,0], 1[0,1,0], 2[0,0,1] 
# y의 라벨의 갯수만큼 원핫인코딩의 벡터값이 부여된다
## 다중분류시에는 y에 원핫인코딩 필수!!!
#cf) 임베딩

2. softmax
# - 노드의 값의 합을 1로 만들어주는 활성화 함수
# 소프트맥스를 통해 변환된 값의 합은 무조건 1

3. categorical Cross entropy #분류화 크로스 엔트로피
# - 1을 가지는 것에 대해서만 오차값을 처리하여 오차값을 합해서 평균을 낸다.

## 회귀와 분류

"""

from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

# 모델
data = load_wine()
x = data.data #(178, 13)
y = data.target #(178, 3)

print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48

y = pd.get_dummies(y)
print(y)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=4343,  stratify=y)

####스케일링 적용######
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.fit_transform(x_test)


#모델 구성
from keras.layers import Dropout

model = Sequential()
model.add(Dense(64, input_dim = 13))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(3, activation='softmax'))

#컴파일 및 훈련
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode = 'auto', 
    patience = 20, 
    verbose=1, 
    save_best_only=True, 
    filepath=".//_save//mcp2//keras30_09_save_wine.hdf5"
)

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=100,
    restore_best_weights=True
)
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.2, callbacks=[es, mcp])

#평가 및 예측
loss = model.evaluate(x_test, y_test)
result = np.round(model.predict([x]))
y_pre = np.round(model.predict([x_test]))
acc = accuracy_score(y_test, y_pre)

print("로스", loss, "정확도", acc)
# 로스 [0.0024288140702992678, 1.0] 정확도 1.0
# 로스 [2.1258176730043488e-06, 1.0] 정확도 1.0
# 로스 [0.10237397253513336, 0.9722222089767456] 정확도 0.9722222222222222
# 로스 [1.4768315850233193e-06, 1.0] 정확도 1.0
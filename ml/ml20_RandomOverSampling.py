import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)


#1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target
# print(x.shape, y.shape) #(178, 13) (178,)
# print(np.unique(y, return_counts = True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48

# 데이터를 증폭하기 위해서, 강제로 불균형 데이터로 만들기
#  
x = x[:-40]
y = y[:-40]
# print(y)
# print(pd.value_counts(y))

# y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=777, 
                                                    stratify=y)

"""
#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim = 13))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 및 예측
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, 
          validation_split=0.3)

#4. 평가 및 예측
result = model.evaluate(x_test, y_test)
print("loss", result[0])
print("acc", result[1])

#f1 
y_pre = np.argmax(model.predict(x_test), axis = 1)  # 3개의 값이 아니까 argmax
f1 = f1_score(y_test, y_pre, average='macro')
print("f1", f1)

# loss 0.5964774489402771
# acc 0.8571428656578064
# f1 0.7988505747126436
"""

####################SMOTE########################
# !pip install imblearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk
# print(sk.__version__) 1.5.1

#전체 데이터를 증폭해도 되지만, 전체를 증폭하는 경우에는 과적합의 문제가 발생한다. 
# 따라서 train data만 증폭한다.

print(np.unique(y_train, return_counts = True)) 
#(array([0, 1, 2]), array([47, 57,  6], dtype=int64))


# data증폭 
# data가 작은 경우에 문제가 없으나, 
# data가 큰 경우에는 증폭에 걸리는 시간이 제곱만큼 증가함
smote = RandomOverSampler(random_state = 777,)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(np.unique(y_train, return_counts = True))
# (array([0, 1, 2]), array([57, 57, 57], dtype=int64))
######################################################

#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_dim = 13))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 및 예측
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, 
          validation_split=0.3)

#4. 평가 및 예측
result = model.evaluate(x_test, y_test)
print("loss", result[0])
print("acc", result[1])

#f1 
y_pre = np.argmax(model.predict(x_test), axis = 1)  # 3개의 값이 아니까 argmax
f1 = f1_score(y_test, y_pre, average='macro')
print("f1", f1)

"""
loss 0.5255564451217651
acc 0.8571428656578064
f1 0.7988505747126436
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
#데이터
x, y = load_digits(return_X_y=True)
print(x.shape, y.shape) #(1797, 64) (1797,)
print(pd.value_counts(y, sort=False))

ohe = OneHotEncoder(sparse=False)
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)
print(x.shape, y.shape) #(1797, 64) (1797, 10)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=6666, 
                                                    stratify=y)

###스케일링 적용###
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)

# 모델 구성
from keras.models import load_model
model = load_model("C:\\ai5\\_save\\mcp2\\keras30_10_save_digits.hdf5")

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pre)

print(loss, acc)

# [2.80710506439209, 0.9527778029441833] 0.9527777777777777
# [104.78390502929688, 0.5361111164093018] 0.5361111111111111
# [3.353447198867798, 0.9416666626930237] 0.9416666666666667
# [24.242084503173828, 0.9222221970558167] 0.9222222222222223
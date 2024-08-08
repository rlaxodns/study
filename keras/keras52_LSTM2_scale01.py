import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],[20,30,40],
              [30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print(x.shape, y.shape)
# (13, 3) (13,)
x_predict = np.array([50, 60, 70])

x = x.reshape(13,3,1)
x_predict = np.array([50, 60, 70]).reshape(1,3,1)


# load_model
model = load_model("C:\\ai5\\_save\\keras52\\keras52_LSTM2_scale02.hdf5")

#4. 평가 및 예측
loss = model.evaluate(x, y)
result = model.predict(x_predict)

print(loss, result)

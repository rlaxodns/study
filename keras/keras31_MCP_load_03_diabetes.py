from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) #(442, 10) (442,)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=725)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

#2. 모델 불러오기
from keras.models import load_model
model = load_model("C:\\ai5\\_save\\mcp2\\keras30_03_save_diabets.hdf5")

#4. 에측 밒 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict([x_test])
r2 = r2_score(y_test, y_predict)

print(loss)
print(r2)

# print(hist.history)

"""
random_state=72
epochs=500, batch_size=1
2212.451171875
0.6306960472982034

2251.953857421875
0.6241022268402279

<적용 후>
2828.406005859375
0.5455834051787203

2793.3154296875
0.551221141642545

2755.257568359375
0.5573355571694054
"""


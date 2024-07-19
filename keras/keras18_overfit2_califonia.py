from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,
                                    random_state=6235)

# [실습]r2 059이상
#모델
model = Sequential()
model.add(Dense(100, input_dim = 8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 컴파일 및 훈련
import time
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode = min,
    patience=3,
    restore_best_weights=True)

model.compile(loss='mse', optimizer='adam')
st_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=10, 
          validation_split=0.2, verbose=2, callbacks=[es])
end_time = time.time()

# 예측 및 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict([x_test])

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)

print(loss)
print(r2)
print(end_time-st_time)

"""0.5996702909469604
0.561659386010252

0.6130666732788086
0.5518669095763122

0.6405205726623535
0.5317990030598638
"""

# 시각화
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'], c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], c= 'blue', label = 'val_loss')
plt.grid()
plt.legend(loc = 'upper right')
plt.title('캘리포니아_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

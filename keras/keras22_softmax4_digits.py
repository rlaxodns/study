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


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6666, stratify=y)

# 모델 구성
model = Sequential()
model.add(Dense(128, input_dim = 64))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(10, activation='softmax'))

# 컴파일 및 훈련
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience=100,
    restore_best_weights=True
)
model.compile(loss = 'categorical_crossentropy', optimizer='adam'
              , metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, batch_size=1,
          validation_split=0.2)

# 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_pre)

print(loss, acc)
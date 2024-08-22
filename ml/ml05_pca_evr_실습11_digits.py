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

for i in range(4):
    x = [60,54, 42, 25]
  #pca
    from sklearn.decomposition import PCA
    pca = PCA()
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    # cum = np.cumsum(pca.explained_variance_ratio_)

    # print(np.argmax(cum)+1)  # 60 # 인덱스라서 +1을 해줘야함
    # print(np.argmax(cum>= 0.999)+1) # 54
    # print(np.argmax(cum>= 0.99)+1)  # 42
    # print(np.argmax(cum>= 0.95)+1)  # 25


    # 모델 구성
    model = Sequential()
    model.add(Dense(128, input_dim = x_train1.shape[1]))
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
    model.fit(x_train1, y_train,
              epochs=100,
              batch_size=32,
              validation_split=0.2, 
              verbose=0)

    # 평가 및 예측
    loss = model.evaluate(x_test1, y_test)
    y_pre = np.round(model.predict(x_test1))
    acc = accuracy_score(y_test, y_pre)

    print(loss, acc)

"""
[0.2881384789943695, 0.9444444179534912] 0.9444444444444444
12/12 [==============================] - 0s 2ms/step - loss: 0.3019 - acc: 0.9667
[0.30187374353408813, 0.9666666388511658] 0.9666666666666667
12/12 [==============================] - 0s 2ms/step - loss: 0.3844 - acc: 0.9639
[0.3843536376953125, 0.9638888835906982] 0.9638888888888889
12/12 [==============================] - 0s 1ms/step - loss: 0.3353 - acc: 0.9694
[0.33531367778778076, 0.9694444537162781] 0.9694444444444444
"""
# # [2.80710506439209, 0.9527778029441833] 0.9527777777777777
# # [104.78390502929688, 0.5361111164093018] 0.5361111111111111
# # [3.353447198867798, 0.9416666626930237] 0.9416666666666667
# # [24.242084503173828, 0.9222221970558167] 0.9222222222222223
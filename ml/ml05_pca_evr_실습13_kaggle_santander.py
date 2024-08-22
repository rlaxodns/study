# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time

# 데이터 구성
path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"
train = pd.read_csv(path+"train.csv", index_col=0)
test = pd.read_csv(path+"test.csv", index_col=0)
sub = pd.read_csv(path+"sample_submission.csv", index_col = 0)

x = train.drop(['target'], axis = 1)
y = train['target']
print(pd.value_counts(y))
print(x.shape,y.shape) #(200000, 200) (200000,)

# y = pd.get_dummies(y)
# print(y.shape) #(200000, 2)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                                    random_state=4343)

####스케일링 적용####
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)
test = rbs.transform(test)

for i in range(3):
    x = [200, 198, 189]
#pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

# cum = np.cumsum(pca.explained_variance_ratio_)

# print(np.argmax(cum)+1)  # 13 # 인덱스라서 +1을 해줘야함
# print(np.argmax(cum>= 0.999)+1) # 13
# print(np.argmax(cum>= 0.99)+1)  # 12
# print(np.argmax(cum>= 0.95)+1)  # 9

    # 모델
    model = Sequential()
    model.add(Dense(64, input_dim = x_train1.shape[1]))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(1, activation='sigmoid'))

    # 컴파일 및 훈련
    es = EarlyStopping(
        monitor='val_loss',
        mode = 'min',
        patience=50,
        restore_best_weights=True
    )
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
    model.fit(x_train1, y_train, epochs=1, batch_size=256, 
              validation_split= 0.2, callbacks=[es], verbose=0)

    # 평가 및 예측
    loss = model.evaluate(x_test1, y_test)
    # result = np.round(model.predict(test))
    y_pre = np.round(model.predict(x_test1))
    acc = accuracy_score(y_test, y_pre)

    print(loss, acc)
"""
[0.23761069774627686, 0.9121500253677368] 0.91215
1250/1250 [==============================] - 2s 2ms/step - loss: 0.2381 - acc: 0.9115
[0.2381187528371811, 0.9114500284194946] 0.91145
1250/1250 [==============================] - 3s 2ms/step - loss: 0.2376 - acc: 0.9111
[0.23763839900493622, 0.9110749959945679] 0.911075"""
# sub['target'] = result
# sub.to_csv(path +"sample_submission03.csv")

# # [0.23117949068546295, 0.9138249754905701] 0.913825
# # [0.23127810657024384, 0.9140750169754028] 0.914075
# # [0.23467880487442017, 0.9129999876022339] 0.913
# # [0.23132435977458954, 0.9133999943733215] 0.9134
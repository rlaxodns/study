# https://dacon.io/competitions/official/236068/overview/description

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

#1. 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\train.csv", index_col = 0)
test = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\test.csv", index_col=0)
submission = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\sample_submission.csv", index_col = 0)

print(train.info(), train.shape) #None (652, 9)
print(train.columns)
"""Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')"""

x = train.drop(['Outcome'], axis=1)
y = train['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
                                                   shuffle=True, random_state=4343)
# print(pd.DataFrame(x).value_counts)

###스케일링 적용####
from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)

for i in range(2):
    x = [8, 7]

###pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

# cum = np.cumsum(pca.explained_variance_ratio_)

# print(np.argmax(cum)+1)  # 8 # 인덱스라서 +1을 해줘야함
# print(np.argmax(cum>= 0.999)+1) # 8
# print(np.argmax(cum>= 0.99)+1)  # 7
# print(np.argmax(cum>= 0.95)+1)  # 7



    #2. 모델 구성
    model =Sequential()
    model.add(Dense(64, input_dim = x_test1.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일 및 훈련
    es = EarlyStopping(
        monitor='val_loss',
        mode ='min',
        patience= 200,
        restore_best_weights=True
    )
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
    model.fit(x_train1, y_train, epochs = 100, batch_size=32,
            validation_split=0.2, callbacks=[es], verbose=0)

    #4. 평가 및 예측
    loss = model.evaluate(x_test1, y_test)
    y_pre = np.round(model.predict([x_test1]))

    # result = np.round(model.predict([test]))

    acc = accuracy_score(y_test, y_pre)
    # print(y_pre)
    print(loss)
    print(acc)

"""
[4.460745334625244, 0.7175572514533997]
0.7175572519083969
5/5 [==============================] - 0s 2ms/step - loss: 3.9335 - acc: 0.7176
[3.933464765548706, 0.7175572514533997]
0.7175572519083969
"""
# submission['Outcome'] = result
# submission.to_csv("C:\\ai5\\_data\\dacon\\diabetes\\sample_submission1555.csv")

# """
# [0.5546854734420776, 0.7099236845970154]
# 0.7099236641221374

# [0.5881093740463257, 0.694656491279602]
# 0.6946564885496184

# [0.5346061587333679, 0.7633587718009949]
# 0.7633587786259542

# <적용 후>
# [0.4714379608631134, 0.7786259651184082]
# 0.7786259541984732

# [0.5277946591377258, 0.7633587718009949]
# 0.7633587786259542
# """
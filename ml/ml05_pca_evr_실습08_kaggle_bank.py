# train['종가'] = train['종가'].str.replace(',', '')
# https://www.kaggle.com/c/playground-series-s4e1/data
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping

# 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\train.csv", index_col=[0, 1, 2])
test = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\test.csv", index_col= [0, 1, 2])
submission = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\sample_submission.csv", index_col=[0])

print(train.dtypes)
"""Surname             object
CreditScore          int64
Geography           object
Gender              object
Age                float64
Tenure               int64
Balance            float64
NumOfProducts        int64
HasCrCard          float64
IsActiveMember     float64
EstimatedSalary    float64
Exited               int64"""
# France = 1, Spain = 2, Germany = 3 // Male = 1, Female =2

# train['Geography'] = train['Geography'].str.replace('France', '1')
# train['Geography'] = train['Geography'].str.replace('Spain', '2')
# train['Geography'] = train['Geography'].str.replace('Germany', '3')

# train['Gender'] = train['Gender'].str.replace('Male', '1')
# train['Gender'] = train['Gender'].str.replace('Female', '2')

# train = train.astype(float)
# print(train.dtypes)

# test['Geography'] = test['Geography'].str.replace('France', '1')
# test['Geography'] = test['Geography'].str.replace('Spain', '2')
# test['Geography'] = test['Geography'].str.replace('Germany', '3')

# test['Gender'] = test['Gender'].str.replace('Male', '1')
# test['Gender'] = test['Gender'].str.replace('Female', '2')
# test = test.astype(float)
# print(test.dtypes)

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# train['Geography'] = le.fit_transform(train['Geography'])
# train['Gender'] = le.fit_transform(train['Gender'])
# test['Geography'] = le.fit_transform(test['Geography'])
# test['Gender'] = le.fit_transform(test['Gender'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train['Geography'] = le.fit_transform(train['Geography'])
train['Gender'] = le.fit_transform(train['Gender'])

test['Geography'] = le.fit_transform(test['Geography'])
test['Gender'] = le.fit_transform(test['Gender'])

x = train.drop(['Exited'], axis = 1)
y = train['Exited']

print(x.shape, y.shape) #(165034, 10) (165034,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
                                                    random_state= 6666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)

for i in range(2):
    x =[10, 8]
##pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

# cum = np.cumsum(pca.explained_variance_ratio_)

# print(np.argmax(cum)+1)  # 10 # 인덱스라서 +1을 해줘야함
# print(np.argmax(cum>= 0.999)+1) # 10
# print(np.argmax(cum>= 0.99)+1)  # 10
# print(np.argmax(cum>= 0.95)+1)  # 8



    #2. 모델 구성
    model = Sequential()
    model.add(Dense(500, input_dim = x_train1.shape[1], activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. 컴파일 및 훈련
    es = EarlyStopping(
        monitor='val_loss',
        mode = 'min',
        patience=20,
        restore_best_weights=True
    )
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
    model.fit(x_train1, y_train, epochs = 100, batch_size=1024, verbose=0,
              validation_split=0.3, callbacks=[es])
    # validation은 훈련에 있어서 미미하게 영향을 미친다

    #4. 평가 및 훈련
    loss = model.evaluate(x_test1, y_test)
    y_pre = np.round(model.predict([x_test1]))
    # result = np.round(model.predict([test]))

    acc = accuracy_score(y_test, y_pre)

    print(loss, acc)

"""
[0.3291511833667755, 0.8637561798095703] 0.8637561729330142
1032/1032 [==============================] - 2s 2ms/step - loss: 0.3352 - acc: 0.8592
[0.3351631164550781, 0.8592116832733154] 0.8592116823704062
"""

# submission['Exited'] = result
# submission.to_csv("C:\\ai5\\_data\\kaggle\\bank\\sample_submission_153311.csv")

# # [0.3274446725845337, 0.8632714152336121] 0.8632714272730027
# # [0.32849907875061035, 0.8626654744148254] 0.8626654951979883
# # [0.3284316956996918, 0.8633320331573486] 0.8633320204805042
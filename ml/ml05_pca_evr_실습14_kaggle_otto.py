import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 데이터 구성
path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sub = pd.read_csv(path + "sampleSubmission.csv", index_col=0, )

le = LabelEncoder()
train["target"] = le.fit_transform(train["target"])

x = train.drop(['target'], axis=1)
y = train['target']


y = pd.get_dummies(y)
print(x.shape, y.shape) #(61878, 93) (61878, 9)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=7777, stratify=y)

####스케일링 적용####
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)
test = rbs.transform(test)

for i in range(4):
    x = [93, 90, 80 ,58]

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

    #. 모델 구성
    model = Sequential()
    model.add(Dense(128, input_dim = x_train1.shape[1], activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(9, activation='softmax'))

    # 컴파일 및 훈련
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=32,
        restore_best_weights=True
    )
    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',
                  metrics=['acc'])
    model.fit(
        x_train1,
        y_train,
        callbacks=[es],
        epochs=10,
        batch_size=256,    
        validation_split=0.3,
        verbose=0
        )

    # 평가 및 예측
    loss = model.evaluate(x_test1, y_test)
    # result = model.predict(test)
    y_pre = model.predict(x_test1)


    print(loss[0], loss[1])

"""
387/387 [==============================] - 1s 3ms/step - loss: 0.6460 - acc: 0.7787
0.6459848284721375 0.7786845564842224
387/387 [==============================] - 1s 3ms/step - loss: 0.5887 - acc: 0.7891
0.5887365937232971 0.7891079783439636
387/387 [==============================] - 1s 3ms/step - loss: 0.5802 - acc: 0.7818
0.5801680684089661 0.7818357944488525
387/387 [==============================] - 1s 3ms/step - loss: 0.6108 - acc: 0.7746
0.6107622981071472 0.7745636701583862
"""

# sub[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8",
#      "Class_9"]] = result
# sub.to_csv(path + "sub0011.csv")

# # 0.5722733736038208 0.793228805065155
# # 0.6055837869644165 0.7788461446762085
# # 0.5790331959724426 0.7954912781715393
# # 0.5654720664024353 0.7845022678375244
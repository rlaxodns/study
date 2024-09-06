import numpy as np
from keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


##스케일링
mms = MinMaxScaler()
x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le2 = LabelEncoder()
optimizers = ['adam', 'rmsprop', 'adadelta']
activation = ['relu', 'elu', 'selu', 'linear']

optimizers = le.fit_transform(optimizers)
activation = le2.fit_transform(activation)

# print(optimizers, activation) #[1 2 0] [2 0 3 1]

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# print(x_train.shape, y_train.shape) #(60000, 784) (60000,)

import pandas as pd
#1-2. get_dummies
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델
bayesian_params = {
    'batchs' : (1, 64),
    'optimizers' : (0, 2),
    'dropout' : (0.1, 0.5),
    'activation' : (0, 3),
    'node1' : (1, 264),
    'node2' : (1, 512),
    'node3' : (1, 512),
    'node4' : (1, 1024),
    'epochs': (100, 1264)
    }

from sklearn.metrics import r2_score, accuracy_score
def build_model (dropout=0.2, optimizers='adam', activation='relu', 
                 node1=256, node2=128, node3=64, node4=16, epochs=100,
                 batchs = 16):
    
    batchs = int(np.trunc(batchs))
    optimizers = le.inverse_transform([int(np.trunc(optimizers))])[0]
    dropout = dropout
    activation = le2.inverse_transform([int(np.trunc(activation))])[0]
    node1 = int(np.trunc(node1))
    node2 = int(np.trunc(node2))
    node3 = int(np.trunc(node3))
    node4 = int(np.trunc(node4))
    epochs = int(np.trunc(epochs))

    input = Input(shape=(28, 28, 1))
    x = Conv2D(node3,(3,3), activation=activation)(input)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(dropout)(x)
    x = Dense(node1, activation=activation)(x)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(dropout)(x)
    x = Dense(node3, activation=activation)(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs= input, outputs = output)

    model.compile(loss = 'categorical_crossentropy', metrics=['acc'],
                  optimizer=optimizers)

    y_pre = model.predict(x_test)
    result = accuracy_score(y_test, np.round(y_pre))

    return result


bay = BayesianOptimization(
    f = build_model,
    pbounds=bayesian_params,
    random_state=333,
)

import time
n_iter = 10
st = time.time()
bay.maximize(init_points=5,
             n_iter=n_iter)
et = time.time()

print(bay.max)
print(n_iter, "걸린 시간", round(et-st, 2))
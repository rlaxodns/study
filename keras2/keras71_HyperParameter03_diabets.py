import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split


#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2887,)

print(x_train.shape, y_train.shape) #(353, 10) (353,)

#2. 모델
def build_model(drop=0.5, optimizer = 'adam', activation = 'relu',
                node1 = 128, node2 = 64, node3 = 32, 
                node4 = 16, node5 = 8, lr = 0.001):
    input = Input(shape = (10,))
    x = Dense(node1, activation=activation)(input)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation)(x)
    x = Dense(node5, activation=activation)(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs = input, outputs = output)

    model.compile(optimizer=optimizer, metrics=['mae'],
                  loss = 'mse')
    return model

def create_hypermeters():
    batchs = [16, 32, 8, 4, 2]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.2, 0.3, 0.4, 0.5]
    activation = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16, 8]

    return{'batch_size': batchs,
           'optimizer': optimizers,
           'drop':dropout,
           'activation':activation,
           'node1':node1,
           'node2':node2,
           'node3':node3, 
           'node4':node4, 
           'node5': node5}

hyperparameters = create_hypermeters()
print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn = build_model, verbose = 1)
# 사이킷런에서 케라스 모델을 인식할 수 있도록 래핑하는 과정

model = RandomizedSearchCV(keras_model, hyperparameters, cv = 5,
                           n_iter=10,  #기본값은 10
                            
                           verbose=1)

import time
st = time.time()
model.fit(x_train, y_train, epochs = 1000)
et = time.time()

print("걸린시간", round(et-st, 2))
print("params",model.best_params_)
print("estimators", model.best_estimator_)
print('최고점수', model.best_score_)
print("model.score", model.score(x_test, y_test))
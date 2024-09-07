import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.unique(y_train, return_counts = True))


#1-1. 스케일링
x_train = x_train/255.
x_test = x_test/255.
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
# print(x_train.shape, y_train.shape) #(60000, 784) (60000,)

#1-2. get_dummies
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델 구성
def build_model(drop = 0.5, optimizer = 'adam', activation = 'relu',
                node1 = 128, node2 = 64, node3=32, lr = 0.001):
    input = Input(shape = (x_train.shape[1]),)
    x = Dense(node3, activation=activation)(input)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node1, activation=activation)(x)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation)(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs= input, outputs = output)

    model.compile(loss = 'categorical_crossentropy', metrics=['acc'],
                  optimizer=optimizer)
    
    return model


def create_parameters():
    batchs = [1,2,4,8,32,64,128]
    activations = ['relu', 'elu', 'selu', 'linear']
    dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16, 8]

    return {'batch_size':batchs,
            'activation':activations,
            'drop':dropouts,
            'optimizer':optimizers,
            'node1':node1,
            'node2':node2,
            'node3':node3,}

hyperparameters = create_parameters()

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

keras_model = KerasClassifier(build_fn=build_model, verbose = 1)

model = RandomizedSearchCV(keras_model,
                           hyperparameters,
                           cv = 3,
                           n_iter=5,
                        #    n_jobs=-1,
                           verbose = 1)


import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(
    monitor='loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)
rlr = ReduceLROnPlateau(
    monitor='loss',
    mode = 'min',
    patience=30,
    factor=0.09
)

st = time.time()
model.fit(x_train, y_train, epochs = 100,
          callbacks = [es, rlr])
et = time.time()

print("걸린시간", round(et-st, 2))
print("params",model.best_params_)
print("estimators", model.best_estimator_)
print('최고점수', model.best_score_)
print("model.score", model.score(x_test, y_test))
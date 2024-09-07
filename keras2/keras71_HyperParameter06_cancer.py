import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split

#1. 데이터 구성
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=2887,
                                                    stratify=y)

# print(x_train.shape, y_train.shape) #(455, 30) (455,)
# print(np.unique(y_train, return_counts = True)) 
#(array([0, 1]), array([170, 285], dtype=int64))

#2. 모델 구성
def build_model(drop = 0.5, optimizer = 'adam', activation = 'relu',
                node1 = 128, node2 = 64, node3=32, lr = 0.001):
    input = Input(shape = (30,))
    x = Dense(node3, activation=activation)(input)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node1, activation=activation)(x)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs= input, outputs = output)

    model.compile(loss = 'binary_crossentropy', metrics=['acc'],
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
# print(hyperparameters)

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
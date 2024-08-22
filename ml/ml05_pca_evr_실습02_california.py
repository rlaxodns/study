from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,
                                    random_state=6235)

##스케일링 적용##
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()
x_train = mas.fit_transform(x_train)
x_test = mas.transform(x_test)

print(x_train.shape) #(18576, 8

results = []
for i in range(5):
    x = [8, 1, 6, 4, 3]
# ##pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=x[i])
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

# cum = np.cumsum(pca.explained_variance_ratio_)

# print(np.argmax(cum>= 1.0 )+1)  # 1 # 인덱스라서 +1을 해줘야함
# print(np.argmax(cum>= 0.999)+1) # 6
# print(np.argmax(cum>= 0.99)+1)  # 4
# print(np.argmax(cum>= 0.95)+1)  # 3


# [실습]r2 059이상
#모델
    model = Sequential()
    model.add(Dense(100, input_dim = x_train1.shape[1]))
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(1))

    # 컴파일 및 훈련
    import time
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(
        monitor= 'val_loss',
        mode = min,
        patience=3,
        restore_best_weights=True)

    model.compile(loss='mse', optimizer='adam')
    st_time = time.time()
    hist = model.fit(x_train1, y_train, epochs=50, batch_size=10, 
            validation_split=0.2, verbose=2, callbacks=[es])
    end_time = time.time()

    # 예측 및 평가
    loss = model.evaluate(x_test1, y_test)
    y_predict = model.predict(x_test1)

    from sklearn.metrics import r2_score

    r2 = r2_score(y_test, y_predict)

    results.append((i+1, x[i], r2, loss))

for result in results:
    print({result[0]}, {result[1]}, "r2",{result[2]}, "loss", {result[3]})



    """0.5996702909469604
    0.561659386010252

    0.6130666732788086
    0.5518669095763122

    0.6405205726623535
    0.5317990030598638

    <스케일링 적용 후>
    0.5360893607139587
    0.5916927408072075

    0.5275655388832092
    0.5981848140146168
    4.71520209312439
    """


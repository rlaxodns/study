
from tensorflow.keras.models import Sequential
from keras.layers import Dense
import sklearn as sk
print(sk.__version__)  #0.24.2
import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

#1. 데이터 (정규화 과정을 포함)
dataset = load_boston() 
# print(dataset.shape)
print(dataset.DESCR)  # sklearn에서 .describe()와 동일한 데이터의 평균 등을 설명하는 함수
print(dataset.feature_names)  #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
                              #   'B' 'LSTAT']

x = dataset.data
y = dataset.target  #x와 y를 데이터 상에서 분리
# print(x.shape) #(506, 13)
# print(y.shape) #(506,)

x_train, x_test, y_train, y_test =train_test_split(x, y, test_size = 0.2,
                             shuffle=True, random_state=6265)  

#####정규화(07/25)#####
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test) 

for i in range(4):
    x = [13, 12, 8, 4]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=x[i])

    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)

    cumsum = np.cumsum(pca.explained_variance_ratio_)

# print(np.argmax(cumsum>= 1.0)+1)   # 13 # 인덱스라서 +1을 해줘야함
# print(np.argmax(cumsum>= 0.999)+1) # 12
# print(np.argmax(cumsum>= 0.99)+1)  # 8
# print(np.argmax(cumsum>= 0.95)+1)  # 4



# #2. 모델
    model = Sequential()
    # model.add(Dense(100, input_dim = 13))
    model.add(Dense(100, input_shape = (x_train1.shape[1],))) # 이미지의 input_shape = (8,8,1)
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(100))
    model.add(Dense(1))

    #3. 컴파일 및 훈련
    model.compile(loss = 'mse', optimizer='adam')
    start_time = time.time()
    hist = model.fit(x_train1, y_train, epochs=500, batch_size=16,
              validation_split=0.2, verbose=1)
    # hist라고 지정하면서 모델의 훈련에 대한 과정을 입력


    end_time = time.time()

    #4. 예측 및 평가
    loss = model.evaluate(x_test1, y_test)
    y_predict = model.predict(x_test1)
    # result = model.predict([x])


    r2 = r2_score(y_test, y_predict)

    print("훈련시간", round(end_time-start_time, 2), "초")
    print("오차값", loss)
    print("결정계수", r2)

# """<정규화 전>
# 훈련시간 9.87 초
# 오차값 31.81736946105957
# 결정계수 0.6289244398474927

# <정규화 후>
# 훈련시간 9.92 초
# 오차값 24.67317008972168
# 결정계수 0.7122449369631706

# <분리 후 진행>
# 훈련시간 10.7 초
# 오차값 22.435144424438477
# 결정계수 0.6681070297995211

# 훈련시간 11.0 초
# 오차값 22.399995803833008
# 결정계수 0.6686269577326496

# 훈련시간 11.37 초
# 오차값 23.1861515045166
# 결정계수 0.656996965697054

# 훈련시간 10.7 초
# 오차값 22.325042724609375
# 결정계수 0.6697358247730445
# """
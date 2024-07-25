#18-1 카피

# 과적합의 해결책 보다는 과적합의 구간을 찾아보는 과정
# loss값을 줄이다가 훈련이 진행이 되지 않는 과정
# 즉, 성능개선이 없는 구간을 과적합 구간이라고 한다.

# 보스턴의 데이터를 다운받아서 보스턴의 집값 예측

from tensorflow.keras.models import Sequential
from keras.layers import Dense
import sklearn as sk
print(sk.__version__)  #0.24.2
import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

######07/25 _ input_shape_
# 우리가 지금까지 인풋을 입력하면서 input_dim을 활용해왔는데, 입력하는 차원은 (열, 컬럼, 특성, 피쳐)
# input_dim의 경우에는 2차원의 데이터만 입력할 수 밖에 없다 
# 따라서 이미지의 경우에는 2차원으로 정렬하면 데이터의 양이 기하급수적으로 늘어나는 문제가 발생한다
# ex) CNN의 경우에는 다차원으로 입력되어 연산을 한기 때문에 input_dim으로 입력하면
# 데이터를 2차원으로 정렬해서 입력하는 복잡함이 발생한다
# 이를 위해 'input_shape'를 활용
######

#1. 데이터
dataset = load_boston() 
# print(dataset.shape)
print(dataset.DESCR)  # sklearn에서 .describe()와 동일한 데이터의 평균 등을 설명하는 함수
print(dataset.feature_names)  #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
                              #   'B' 'LSTAT']


x = dataset.data
y = dataset.target  #x와 y를 데이터 상에서 분리


print(x.shape) #(506, 13)
print(y.shape) #(506,)


x_train, x_test, y_train, y_test =train_test_split(x, y, test_size = 0.3,
                             shuffle=True, random_state=6265)       

#2. 모델
model = Sequential()
# model.add(Dense(100, input_dim = 13))
model.add(Dense(100, input_shape = (13,))) # 이미지의 input_shape = (8,8,1)
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=16,
          validation_split=0.2, verbose=1)
# hist라고 지정하면서 모델의 훈련에 대한 과정을 입력


end_time = time.time()

#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict([x])


r2 = r2_score(y_test, y_predict)

print("훈련시간", round(end_time-start_time, 2), "초")
print("오차값", loss)
print("결정계수", r2)


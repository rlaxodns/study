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
model.add(Dense(100, input_dim = 13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer='adam')
start_time = time.time()

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode = min, # 모르면 auto, 정확도는 자동으로 최대값 // 기본값은 auto
    patience = 10, #참을성이 적으면 좋은 로스값을 얻을 수 없다.
    # 35.403099060058594
    restore_best_weights=True, # 실제적으로 연산은 멈춘 지점에서 가중치가 잡히기 때문에
    #   최소값 지점으로 가중치로 잡아준다. 
)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=16,
          validation_split=0.2, verbose=16, callbacks = [es])
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
print(hist.history)

"""
#훈련에 대한 딕셔너리
print("=============hist==================")
print(hist) # <keras.callbacks.History object at 0x000002332616E220>
print("=============hist.history========")
print(hist.history) # 로스의 딕셔너리
print(hist.history['loss']) # 로스에 대한 딕셔너리를 확인
# [3215.612060546875, 470.8392333984375, 125.93729400634766, 95.13848876953125, 80.97518920898438] 
print(hist.history['val_loss']) # validation의 로스에 대한 딕셔너리를 확인
# [359.1191101074219, 433.61151123046875, 57.282737731933594, 100.34498596191406, 59.320037841796875]
"""

##.시각화
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='yellow', label = 'loss') # x값이 지정되지 않은 경우에는 x는 시간 순에 따름
#괄호 안에 marker = '.'을 넣으면 에포지점에 점으로 표시해줌
#laber = 그래프의 이름을 loss라고 지정
# + val_loss 도 추가해보기
plt.plot(hist.history['val_loss'], c='green', label = 'val_loss')
plt.legend(loc='upper right') # label 값이 무엇인지 표현
plt.title('보스턴 Loss') # 그래프의 제목과 x축의 제목, y축의 제목
plt.xlabel('epchs')
plt.ylabel('loss')
plt.grid() # 그래프에 격자를 생성
plt.show()

# 파이썬에서 리스트와 딕셔너리, 튜플을 사용하게 되는데
# 실질적으로 리스트와 딕셔너리를 많이 사용

# 해당 딕셔너리를 가지고 y축은 loss, val_loss를 x축은 epochs를 가지고 그래프화 할 수 있다.

"""
1
random=256
epo=1000
26.968708038330078
0.72588187868526

2
ran =625
epo=1000
17.805763244628906
0.7536866961834869

3.
26.192617416381836
0.6945241458236902
"""

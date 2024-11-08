# 우리가 데이터 처리와 모델 구성에 있어서의 문제는 Overfit(과적합)이다.
# 데이터를 훈련시킬 수록, 레이어를 늘릴 수록,
# 데이터를 늘릴 수록 loss 또는 Accuracy가 갱신되지 않는 문제가 발생한다.

# 이를 해결할 수 있는 방안 중 하나, Dropout
# 전체를 훈련시키는 상황에서 랜덤한 노드를 제외하여 훈련시키는 경우에서 
# 더욱 나은 성능을 보이는 경우가 있다
# 과적합에 뛰어난 성능을 보인다 그러나, 이럴거면 처음부터 노드의 수를 줄일 수 있으나, 
# 드랍아웃의 위치를 랜덤하게 바뀌면서 과적합의 문제를 해결할 수 있다.
# 그러나 모델 구성 및 훈련에 있어서만 적용하게되는 것이고 이외의 평가-예측에서는 적용되지 않는다
# 왜냐하면 모델 그 자체를 평가하는 것이지, 드랍아웃을 적용되는 것은 아니다 
# cf) 텐서플로의 경우에서는 자동으로 평가-예측에서는 적용되지 않도록 지정되어 있지만, 
# pytoch의 경우에서는 함수를 적용해야한다



# 29-5 copy

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
print(dataset.feature_names)  

x = dataset.data
y = dataset.target
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

x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test) 


# 모델 불러오기
from keras.models import load_model
model = load_model(".\\_save\\mcp2\\keras30_0729_1307_0151-12.3255.hdf5")

#4. 예측 및 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict([x])
r2 = r2_score(y_test, y_predict)

print("오차값", loss)
print("결정계수", r2)

# 오차값 22.29594612121582
# 결정계수 0.6701661973511581

# train_test_split 후 스케일링 후 pca

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # dicision tree의 앙상블
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



#1. 데이터
data = load_iris()
x = data['data']
y = data.target
# print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.3,
                                                    random_state=123,
                                                    stratify=y, # y의 라벨에 맞춰서 분류의 갯수 맞춰준다 
                                                    shuffle=True) # shuffle = False가 디폴트


# 일반적으로 스케일링 후 pca해야 성능이 좋음
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
# pca를 사용시에 스케일링도 같이 사용

for i in range(x.shape[1]-1,1,-1):
    i


pca = PCA(n_components=i)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
# print(x)
print(x_train.shape, x_test.shape)


#2. 모델 구성
model = RandomForestClassifier(random_state=123)
# 모델에 컴파일러까지 통합되어있음

#3. 훈련
model.fit(x_train, y_train) # epochs default 100

#4. 평가 및 예측
result = model.score(x_test, y_test) # accuracy 점수와 동일 # regressor는 r2 score
model.predict(x_test)

print(x.shape)
print(result)

#1 0.8666666666666667
#2 0.8666666666666667 
#3 0.9333333333333333
#4 0.9666666666666667

evr = pca.explained_variance_ratio_ # 설명 가능한 변화율
print(evr)
print(sum(evr))

import numpy as np

evr_cumsum = np.cumsum(evr) # cumsum = 누적합
print(evr_cumsum)

import matplotlib.pyplot as plt

plt.plot(evr_cumsum)
plt.grid()
plt.show()
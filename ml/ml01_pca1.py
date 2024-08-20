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


std = StandardScaler()
x = std.fit_transform(x)
# pca를 사용시에 스케일링도 같이 사용

pca = PCA(n_components=3)
x = pca.fit_transform(x)

# print(x)
# print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.4,
                                                    random_state=6465,
                                                    stratify=y, # y의 라벨에 맞춰서 분류의 갯수 맞춰준다 
                                                    )

#2. 모델 구성
model = RandomForestClassifier(random_state=4343)
# 모델에 컴파일러까지 통합되어있음

#3. 훈련
model.fit(x_train, y_train,
          epochs=1000) # epochs default 100

#4. 평가 및 예측
result = model.score(x_test, y_test) # accuracy 점수와 동일 # regressor는 r2 score
model.predict(x_test)

print(x.shape)
print(result)

#1 0.8666666666666667
#2 0.8666666666666667 
#3 0.9333333333333333
#4 0.9666666666666667

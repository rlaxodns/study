from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#1. 데이터
data = load_iris()
x = data['data']
y = data.target
# print(x.shape, y.shape) #(150, 4) (150,)

pca = PCA(n_components=3)
x = pca.fit_transform(x)

std = StandardScaler()
x = std.fit_transform(x)
# pca를 사용시에 스케일링도 같이 사용

print(x)
print(x.shape)
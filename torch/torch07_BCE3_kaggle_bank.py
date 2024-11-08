import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
Device = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, Device)

# 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\train.csv", index_col=[0, 1, 2])
test = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\test.csv", index_col= [0, 1, 2])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train['Geography'] = le.fit_transform(train['Geography'])
train['Gender'] = le.fit_transform(train['Gender'])

test['Geography'] = le.fit_transform(test['Geography'])
test['Gender'] = le.fit_transform(test['Gender'])

x = train.drop(['Exited'], axis = 1)
y = train['Exited']
# print(x.shape, y.shape) #(165034, 10) (165034,)
print(np.unique(y, return_counts=True))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
                                                    random_state= 6666)


#스탠다드 스케일러 적용
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(Device)
x_test = torch.FloatTensor(x_test).to(Device)

y_train = torch.FloatTensor(y_train.values).unsqueeze(1).to(Device)
y_test = torch.FloatTensor(y_test.values).unsqueeze(1).to(Device)
print(x_train.shape, y_train.shape) #torch.Size([132027, 10]) torch.Size([132027, 1])


#2. 모델
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 1),
    nn.Sigmoid()
).to(Device)

#3. 컴파일 및 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()

    hypothesis = model(x)
    loss = criterion(hypothesis, y)

    loss.backward()
    optimizer.step()

    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print("epoch:", epoch, "loss:", loss)

#4. 평가 및 예측
def evaluate(model, criterion, x, y):
    model.eval() # 역전파, 가중치 갱신, 기울기 계산을 할 수 있기도 없기도
                 # dropout, batchnormal 얘네 안함
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)

    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss", loss2)

y_predict = model(x_test)
acc = accuracy_score(y_test.cpu().numpy(), np.round(y_predict.detach().cpu().numpy()))
print("acc", acc)

"""
최종 loss 19.809490203857422
acc 0.8635743933105099
"""
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


#1. 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\train.csv", index_col = 0)
test = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\test.csv", index_col=0)
submission = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\sample_submission.csv", index_col = 0)

x = train.drop(['Outcome'], axis=1)
y = train['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
                                                   shuffle=True, random_state=4343)

#스탠다드 스케일러 적용
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(Device)
x_test = torch.FloatTensor(x_test).to(Device)

y_train = torch.FloatTensor(y_train).unsqueeze(1).to(Device)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(Device)

print(x_train.shape, y_train.shape) #torch.Size([521, 8]) torch.Size([521, 1])
print(x_test.shape, y_test.shape) #torch.Size([131, 8]) torch.Size([131, 1])


#2. 모델
model = nn.Sequential(
    nn.Linear(8, 64),
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
최종 loss 36.53081130981445
acc 0.6335877862595419
"""
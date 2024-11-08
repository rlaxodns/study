import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(USE_CUDA, DEVICE, torch.__version__)

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)
# print(x.shape, y.shape)     # torch.Size([150, 4]) torch.Size([150])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=777,
                                                    stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# print(x_train.shape, y_train.shape) #(120, 4) (120,)
# print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2]), array([40, 40, 40], dtype=int64))

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)


#2. 모델
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
).to(DEVICE)

#3. 컴파일 및 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)

    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 10
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch:{epoch}, loss:{loss}')

#4. 평가, 예측
def evaluate(model, criterion, x_test, y_test):
    model.eval()

    with torch.no_grad():
        y_pred = model(x_test)
        loss2 = criterion(y_pred, y_test)

        return loss2.item()
    
loss2 = evaluate(model, criterion, x_test, y_test)
print(loss2)

## Accuracy Score ##
y_predict = model(x_test)
# acc = accuracy_score(y_test.cpu().numpy(), np.argmax(y_predict.detach().cpu().numpy,
#                                                       axis=1))
# print(acc)
print(y_predict[:5])
"""
tensor([[-1.0706,  1.3722, -0.4002],
        [-1.5250,  1.6615, -0.1350],
        [-0.7954,  0.9185, -0.1386],
        [-2.3440,  1.8099,  0.6130],
        [-2.1302,  1.4679,  0.8583]], device='cuda:0',
       grad_fn=<SliceBackward0>)
"""

y_predict = torch.argmax(model(x_test), dim=1)
print(y_predict[:5])
"""
tensor([1, 1, 1, 2, 2], device='cuda:0')
"""

acc2 = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())
print(acc2)
# 0.9333333333333333

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))
print(f'accuracy : {score:.4f}')
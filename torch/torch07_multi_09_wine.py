import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(DEVICE)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1024,
                                                    stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

print(x_train.shape, y_train.shape) #torch.Size([142, 13]) torch.Size([142])
print(x_test.shape, y_test.shape) #torch.Size([36, 13]) torch.Size([36])

#2. 모델
model = nn.Sequential(
    nn.Linear(13, 36),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(36, 72),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(36, 72),
    nn.ReLU(),
    nn.Linear(72, 36),
    nn.ReLU(),
    nn.Linear(36, 13),
    nn.ReLU(),
    nn.Linear(13, 1)).to(DEVICE)

#3. 컴파일 및 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1e-4)

def train(model, criterion, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)

    loss.backward()
    optimizer.step()

    return loss.item()

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print("epoch", epoch,"loss", loss)

def evaluate(model, criterion, x_train, y_train):
    model.eval()

    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)

        return loss.item()
    
loss2 = evaluate(model, criterion, x_test, y_test)
print("최종loss", loss2)

### acc ###
y_predict = model(x_test)
acc = accuracy_score(y_test.cpu().numpy(), np.argmax(y_predict.detach().cpu().numpy(), axis = 1))
print(acc)
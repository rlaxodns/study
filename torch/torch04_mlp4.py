import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print("torch", torch.__version__, "사용기기", DEVICE)

#1. 데이터 
x = np.array([range(10)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [10,9,8,7,6,5,4,3,2,1]]).T
# print(x.shape, y.shape) (1, 10) (3, 10)
#예측값 10

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)
# print(x.shape, y.shape) torch.Size([10, 1]) torch.Size([10, 3])

#2. model
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Linear(32, 64), 
    nn.Linear(64, 64), 
    nn.Linear(64, 32), 
    nn.Linear(32, 3), 
).to(DEVICE)

#3. 컴파일 및 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    model.train()

    optimizer.zero_grad

    hypothesis = model(x)
    loss = criterion(hypothesis, y)

    loss.backward()
    optimizer.step()

    return loss.item()

epochs = 300
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print(epoch, loss)


#4. 평가 및 예측
def evaluate(model, criterion, x, y):
    model.eval()

    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)

    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print(loss2)

result = model(torch.Tensor([[10]]).to(DEVICE))
print(result.detach().cpu().numpy())

# 69783478272.0
# [[ 664121.    399826.62 -198395.14]]
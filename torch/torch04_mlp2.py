import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print("torch", torch.__version__, "사용기기", DEVICE)

#1. 데이터 
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [10,9,8,7,6,5,4,3,2,1]]).T
y = np.array([1,2,3,4,5,6,7,7,9,10])                
# print(x.shape, y.shape) #(10, 3) (10,)
# 예측값 10, 1.3, 1 

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
# print(x.size(), y.size()) #torch.Size([10, 3]) torch.Size([10])

#2. 모델
model = nn.Sequential(
    nn.Linear(3, 18),
    nn.Linear(18, 36),
    nn.Linear(36, 18),
    nn.Linear(18, 9),
    nn.Linear(9, 1)
    ).to(DEVICE)

#3. 컴파일 및 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()

    hypothesis = model(x)
    loss = criterion(hypothesis, y)

    loss.backward()
    optimizer.step()

    return loss.item()

epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print(epoch, loss)

def evaluate(model, criterion, x, y):
    model.eval()

    with torch.no_grad():
        y_pre = model(x)
        loss2 = criterion(y, y_pre)

    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print("end_loss", loss2)

result = model(torch.Tensor([[10, 1.3, 1]]).to(DEVICE))
print("결과값", result.item())

# end_loss 0.08757402747869492
# 결과값 9.6785306930542
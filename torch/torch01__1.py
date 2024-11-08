import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#1. data
x = np.array([1,2,3])
y = np.array([1,2,3])
print(x.shape, y.shape)

# x = torch.FloatTensor(x) 
# print(x.shape)  # torch.Size([3])
# print(x.size()) # torch.Size([3])

x = torch.FloatTensor(x).unsqueeze(1)
y = torch.FloatTensor(y).unsqueeze(1)
print(x.size(), y.size()) # torch.Size([3, 1]) torch.Size([3, 1])

#2. model
model = nn.Linear(1, 1) #(input, output) #y = xw + b

#3. compile & fit
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    model.train() # 훈련모드
    optimizer.zero_grad() # 각 배치마다 기울기를 초기화하여, 기울기 누적에 의한 문제 해결 # loss를 weight로 미분한 값 = 기울기

    hypothesis = model(x) # y = wx+b
    loss = criterion(hypothesis, y) # loss = mse()

    loss.backward() # 기울기(gradient)값 계산진행, #역전파의 시작
    optimizer.step()  # 가중치 갱신,              #역전파의 끝

    return loss.item()

epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch:{}, loss:{}'.format(epoch, loss)) # verbose

print('==========================================')
# evaluate & predict
def evaluate(model, criterion, x, y):
    model.eval() # 평가모드
    
    with torch.no_grad():
        y_predict = model(x)    
        loss2 = criterion(y, y_predict)

    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print("최종loss,", loss2)

results = model(torch.Tensor([[4]]))
print(results.item())

# 최종loss, 7.768600758344124e-13
# 3.999997854232788
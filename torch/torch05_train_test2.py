import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print("torch", torch.__version__, "사용기기", DEVICE)

x = np.array(range(100))
y = np.array(range(1, 101))
# print(x.shape, y.shape) #(100,) (100,)
x_pred = np.array([101, 102])


x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
x_pred = torch.FloatTensor(x_pred).unsqueeze(1).to(DEVICE)
print(x.shape, y.shape, x_pred.shape) #torch.Size([100, 1]) torch.Size([100, 1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 7777)
# print(x_train.shape, y_train.shape) #torch.Size([70, 1]) torch.Size([70, 1])

#2. model
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Linear(32, 16),
    nn.Linear(16, 8),
    nn.Linear(8, 1),
).to(DEVICE) #(input, output) #y = xw + b

#3. compile & fit
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    # model.train() # 훈련모드
    optimizer.zero_grad() # 각 배치마다 기울기를 초기화하여, 기울기 누적에 의한 문제 해결 # loss를 weight로 미분한 값 = 기울기
                          # 토치의 연산방식이 기울기를 통해서 진행됨
    hypothesis = model(x) # y = wx+b
    
    loss = criterion(hypothesis, y) # loss = mse()

    loss.backward() # 기울기(gradient)값 계산진행, #역전파의 시작
    optimizer.step()  # 가중치 갱신,              #역전파의 끝   # 역전파를 통해서 최적의 가중치를 찾아내는 것

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

results = model(x_pred.to(DEVICE))
print(results.detach().cpu().numpy())

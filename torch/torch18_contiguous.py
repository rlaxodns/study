import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

random.seed(333) # 파이썬 랜덤 고정
np.random.seed(333)
torch.manual_seed(333) #cpu 고정
torch.cuda.manual_seed(333) #gpu 고정

# CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if CUDA else "cpu")

DEVICE = "cuda:0" if torch.cuda.is_available else "cpu"
print(DEVICE)


#1 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3], #각각의 행은 timesteps
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],]
             )

y = np.array([4,5,6,7,8,9,10])

# print(x.shape, y.shape) #(7, 3) (7,) 2차원 형태의 데이터기 때문에 rnn구동 위한 차원 증가의 필요

x = x.reshape(
    x.shape[0],
    x.shape[1], 
    1
) # RNN 사용을 위한 차원 증가
# print(x.shape) #(7, 3, 1) #3D tensor with shape (batch_size, timesteps, features)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)
# print(x.shape, y.size()) #torch.Size([7, 3, 1]) torch.Size([7])

from torch.utils.data import TensorDataset # x,y를 합친다
from torch.utils.data import DataLoader # batch 정의

train_set = TensorDataset(x, y)

train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# aaa = iter(train_loader)
# bbb = next(aaa)
# print(bbb)
# print(bbb[0].size())

#2 model
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.RNN(input_size=1, # 피쳐갯수
                        hidden_size=32, # 아웃풋 노드의 수
                        num_layers=3, # default  // num_layers 조절시 shape이 틀어진다 / 3 or 5가 좋다는 속설
                        batch_first=True # defalt = False 
                       ) #(3, n, 1) --> (n, 3, 1) -<batch_first=True>-> (n, 3, 32)
        
        # self.cell = nn.RNN(1, 32, batch_first=True) 간단하게 가능

        self.fc1 = nn.Linear(3*32, 32) # (n, 3*32) --> (n, 16)
        self.fc2 = nn.Linear(32, 16) # (n, 16) --> (n, 8)
        self.fc3 = nn.Linear(16, 1) # (n, 8) --> (n, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        
        # model.add(LSTM(32, input_shape=(3, 1)))
        x, hidden_state = self.cell(x) 
        # x, _ = self.cell(x) # 히든은 출력을 안하는 경우
        x = self.relu(x)

        # x = x.reshape(-1, 3*32) # view와 reshape의 차이 = 연속적인 것이냐 아니냐의 차이
                                # RNN을 통과시, 대체로 비연속적
                                # view() = 연속적, reshape = 무관

        x = x.contiguous() # 연속적인 형태로 변경 절차
        x = x.view(-1, 3*32)

        x = self.fc1(x)
        # x = self.drop(x)
        x = self.relu(x)

        x = self.fc2(x)
        # x = self.drop(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x
    
model = RNN().to(DEVICE)

from torchsummary import summary
summary(model, (3,1))

#3. 컴파일 및 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4)

def train(model, criterion, optimizer, loader):
    epoch_loss = 0
    
    model.train()

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)

        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)

        loss.backward() # 기울기 계산
        optimizer.step() # 가중치 갱신

        epoch_loss += loss.item() # 가중치 누적

        return epoch_loss / len(loader)
    
def evaluate(model, criterion, loader):
    epoch_loss = 0
    
    model.eval()

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)

            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)

            epoch_loss += loss.item() # 가중치 누적

            return epoch_loss / len(loader)
        
EPOCH = 1000
for epoch in range(1, EPOCH+1):
    loss = train(model, criterion, optimizer, train_loader)

    if epoch % 20 == 0 : 
        print("epoch:{},loss: {}".format(epoch, loss))


#4. 평가 및 예측
x_predict = np.array([[8, 9, 10]])
def predict(model, data):
    model.eval()
    
    with torch.no_grad():
        data = torch.FloatTensor(data).unsqueeze(2).to(DEVICE) # (1, 3)==> (1,3,1)

        y_predict = model(data)

    return y_predict.cpu().numpy()

y_pred = predict(model, x_predict)
print(y_pred)
print("==============================")
print(y_pred[0])
print('==============================')
print(f'{x_predict}의 예측값:{y_pred[0][0]}')

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
                        num_layers=1, # default
                        batch_first=True # defalt = False 
                       ) #(3, n, 1) --> (n, 3, 1) -<batch_first=True>-> (n, 3, 32)
        self.fc1 = nn.Linear(3*32, 16) # (n, 3*32) --> (n, 16)
        self.fc2 = nn.Linear(16, 8) # (n, 16) --> (n, 8)
        self.fc3 = nn.Linear(8, 1) # (n, 8) --> (n, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        
        # model.add(LSTM(32, input_shape=(3, 1)))
        x, hidden_state = self.cell(x) 
        # x, _ = self.cell(x) # 히든은 출력을 안하는 경우
        x = self.relu(x)

        x = x.reshape(-1, 3*32)

        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
    
model = RNN().to(DEVICE)

from torchsummary import summary
summary(model, (3,1))
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
               RNN-1  [[-1, 3, 32], [-1, 2, 32]]               0
              ReLU-2                [-1, 3, 32]               0
            Linear-3                   [-1, 16]           1,552
           Dropout-4                   [-1, 16]               0
              ReLU-5                   [-1, 16]               0
            Linear-6                    [-1, 8]             136
           Dropout-7                    [-1, 8]               0
              ReLU-8                    [-1, 8]               0
            Linear-9                    [-1, 1]               9
================================================================
Total params: 1,697
Trainable params: 1,697
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.05
Params size (MB): 0.01
Estimated Total Size (MB): 0.05
----------------------------------------------------------------
"""
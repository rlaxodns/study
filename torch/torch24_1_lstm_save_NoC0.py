import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import tqdm
from torch.utils.data import DataLoader, Dataset

# Random seed 고정
random.seed(333)
np.random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed(333)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("사용하는 장치:", DEVICE)

# 데이터 불러오기
PATH = './_data/kaggle/netflix/'
train_csv = pd.read_csv(PATH + 'train.csv')

class Custom_Dataset(Dataset):
    def __init__(self, train_csv: pd.DataFrame):
        self.csv = train_csv
        self.x = self.csv.iloc[:, 1:4].values
        # 정규화
        self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x, axis=0) - np.min(self.x, axis=0))
        self.y = self.csv['Close']

    def __len__(self):
        return len(self.x) - 30

    def __getitem__(self, i):
        x = self.x[i: i + 30]
        y = self.y[i + 30]
        return x, y

dataset = Custom_Dataset(train_csv)
train_loader = DataLoader(dataset, batch_size=32)

# LSTM 모델 정의
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=5, batch_first=True)
        self.fc1 = nn.Linear(in_features=30 * 64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x, h0):
        x, (hn, cn) = self.lstm(x, (h0, torch.zeros_like(h0)))
        x = x.contiguous()
        x = x.view(-1, 30 * 64)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = LSTM().to(DEVICE)

# Train 설정
optimizer = optim.Adam(params=model.parameters(), lr=0.001)

for epoch in range(1, 201):
    iterator = tqdm.tqdm(train_loader)

    for x, y in iterator:
        optimizer.zero_grad()
        
        h0 = torch.zeros(5, x.shape[0], 64).to(DEVICE)  # c0 생략

        # 모델 예측 및 손실 계산
        hypothesis = model(x.type(torch.FloatTensor).to(DEVICE), h0)
        loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))

        loss.backward()
        optimizer.step()

        iterator.set_description(f'epoch:{epoch} loss:{loss.item():.10f}')

# 모델 저장
PATH_SAVE = './_save/torch/'
torch.save(model.state_dict(), PATH_SAVE + 't24.pth')

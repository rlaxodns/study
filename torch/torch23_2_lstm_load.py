import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
import tqdm

# Random seed 고정
# random.seed(333)
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

# LSTM 모델 정의 (레이어 이름을 rnn으로 변경)
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=5, batch_first=True)  # hidden_size 조정
        self.fc1 = nn.Linear(in_features=30 * 64, out_features=32)  # hidden_size=64에 맞춰 차원 수정
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x, h0, c0):
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = LSTM().to(DEVICE)

# 모델 평가용 코드
train_loader = DataLoader(dataset, batch_size=1)

y_predict = []
y_test_list = []
total_loss = 0

PATH_SAVE = './_save/torch/'
with torch.no_grad():
    model.load_state_dict(torch.load(PATH_SAVE + 't23.pth', map_location=DEVICE, ))

    for x_test, y_test in train_loader:
        h0 = torch.zeros(5, x_test.shape[0], 64).to(DEVICE)  # num_layers=5, batch_size=1, hidden_size=64
        c0 = torch.zeros(5, x_test.shape[0], 64).to(DEVICE)

        y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE), h0, c0)
        y_predict.append(y_pred.item())
        y_test_list.append(y_test.item())

        loss = nn.MSELoss()(y_pred, y_test.type(torch.FloatTensor).to(DEVICE))
        total_loss += loss.item()



print('Total Loss:', total_loss / len(train_loader))
print("R^2 Score:", r2_score(y_test_list, y_predict))

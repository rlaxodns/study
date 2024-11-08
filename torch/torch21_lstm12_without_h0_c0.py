import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from torchsummary import summary
from torch.utils.data import TensorDataset # x, y를 합친다
from torch.utils.data import DataLoader # batch 정의

random.seed(333)
np.random.seed(333)
torch.manual_seed(333) # torch 고정
torch.cuda.manual_seed(333) # cuda 고정

# RNN과 LSTM의 차이는 c0(cell state)가 추가된 것 뿐이다

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

# print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(DEVICE)

#1 data
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x = np.array(
   [[1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9]])

y = np.array([4, 5, 6, 7, 8, 9, 10])

x = x.reshape(x.shape[0], x.shape[1], 1)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)

print(x.shape, y.size()) # torch.Size([7, 3, 1]) torch.Size([7])

train_set = TensorDataset(x, y)

train_loader = DataLoader(train_set, batch_size = 2, shuffle = True)

# model
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cell = nn.LSTM(1, # number of expected features
                           32, # number of features
                           batch_first = True, # (3, N, 1) -> (N, 3, 1) -> (N, 3, 32)
                           num_layers = 1, # according to legend -> default or 3 or 5 -> good
                           bidirectional = False,
                        #    nonlinearity = 'relu',
                        #    dropout = 0.5,
        ) # model.add(SimpleRNN(32, input_shaper = (3, 1)))

        self.fc1 = nn.Linear(3 * 32, 16) # (N, 3 * 32) -> (N, 16) # bidirectional은 * 2를 해주어야 한다
        self.fc2 = nn.Linear(16, 8) # (N, 16) -> (N, 8)
        self.fc3 = nn.Linear(8, 1) # (N, 8) -> (N, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # if h0 is None:

        # x, hidden_state = self.cell(x) # hidden_state -> rnn에서의 hidden output
        x, _ = self.cell(x)

        x = self.relu(x)

        x = x.contiguous() # this is theoretically better
        # x = x.reshape(-1, 3 * 32)
        x = x.view(-1, 3 * 32) # bidirectional은 * 2를 해주어야 한다

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

model = LSTM().to(DEVICE)

# summary(model, (3, 1))

# train
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = 1e-4)

def train(model, criterion, optimizer, loader):
    epoch_loss = 0

    model.train() # 역전파되고 가중치가 갱신되고 기울기가 계산된다

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)

        optimizer.zero_grad()

        hyperthesis = model(x_batch)

        loss = criterion(hyperthesis, y_batch)

        # 순전파
        # ---------------------------------
        # 역전파

        loss.backward() # 기울기 계산

        optimizer.step() # 가중치 갱신

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def evaluate(model, criterion, loader):
    epoch_loss = 0

    model.eval() # 역전파안함 가중치가 안갱신되고 기울기는 계산할 수 있다

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE).float().view(-1, 1)

            hyperthesis = model(x_batch)

            loss = criterion(hyperthesis, y_batch)

            # 순전파
            # ---------------------------------
            # 역전파

            epoch_loss += loss.item()

    return epoch_loss / len(loader)

for epoch in range(1000):
    loss = train(model, criterion, optimizer, train_loader)

    if epoch % 20 == 0:    
        print('epoch :{}, loss : {:.10f}'.format(epoch, loss))

#4 predict
x_predict = np.array([[8, 9, 10]])

def predict(model, data):
    model.eval()

    with torch.no_grad():
        data = torch.FloatTensor(data).unsqueeze(2).to(DEVICE) # (1, 3) -> (1, 3, 1)

        y_predict = model(data)

    return y_predict.cpu().numpy()

y_predict = predict(model, x_predict)

print(y_predict)
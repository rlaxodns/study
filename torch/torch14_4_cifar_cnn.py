import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import FashionMNIST, CIFAR10
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch :', torch.__version__, '사용 DEVICE :', DEVICE)

""" 정규화 적용하기 """
import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5,), (0.5,))])

path = './study/torch/_data'
train_dataset = CIFAR10(path, train = True, download=True, transform=transf)
test_dataset = CIFAR10(path, train = False, download=True, transform=transf)

print(train_dataset[0][0])
print(test_dataset[0][0])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

### shape 확인 ###
bbb = iter(train_dataloader)
aaa = next(bbb)
print(aaa[0].shape) # torch.Size([32, 3, 56, 56]) (배치, 채널, 넓이, 높이)


#2. 모델
class CNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1), # n, 64, 54, 54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), # n, 64, 27, 27
            nn.Dropout(0.2)
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1,1), stride=1), # n, 32, 27, 27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), # n, 32, 13, 13
            nn.Dropout(0.2)
        )
        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1), # n, 32, 11, 11
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), # n, 16, 5, 5
            nn.Dropout(0.2)
        )
        self.dense_layer1 = nn.Linear(16*5*5, 64)
        self.Drop = nn.Dropout(0.2)
        self.dense_layer2 = nn.Linear(64, 32)
        self.dense_output = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense_layer1(x)
        x = self.Drop(x)
        x = self.dense_layer2(x)
        x = self.dense_output(x)

        return x

model = CNN(3).to(DEVICE)

#3. 컴파일 및 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1e-4)

def train(model, criterion, optimizer, loader):
    model.train()

    epoch_loss = 0
    epoch_acc = 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        hypothesis = model(x_batch)

        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()

        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        return epoch_loss / len(loader), epoch_acc / len(loader)
    
def evaluate(model, criterion, loader):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)

            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()

            epoch_acc += acc.item()
            epoch_loss += loss.item()

        return epoch_loss / len(loader), epoch_acc / len(loader)
    
EPOCH = 100
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_dataloader)
    val_loss, val_acc = evaluate(model, criterion, train_dataloader)
    print("loss", loss, "acc", acc, "val_loss", val_loss, "val_acc", val_acc)

#4. 평가, 예측
et = time.time()
loss, acc = evaluate(model, criterion, test_dataloader)
print("================================================================================")
# print('걸린시간', et - st)
print('최종 Loss :', loss)
print('최종 acc :', acc)

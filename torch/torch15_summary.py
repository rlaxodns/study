import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR100, MNIST
import time


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

""" 정규화 적용해보기 """
import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor(), tr.Normalize((0.5,), (0.5,))])
# min-max(x)-평균(0.5)/표준편차(0.5) = Z-Score Normalization
st = time.time()
#1. data
path = './study/torch/_data'
# train_dataset = MNIST(path, train=True, download=False)
# test_dataset = MNIST(path, train=False, download=False)

train_dataset = MNIST(path, train=True, download=True, transform=transf)
test_dataset = MNIST(path, train=False, download=True, transform=transf)

print(train_dataset[0][0]) #<PIL.Image.Image image mode=L size=28x28 at 0x2ACF36C9850>
print(test_dataset[0][0]) #<PIL.Image.Image image mode=L size=28x28 at 0x2ACB4E78CB0>

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# x_train, y_train = train_dataset.data/255., train_dataset.targets  # 28*28로 롤백
# x_test, y_test = test_dataset.data/255., test_dataset.targets

"""
x_train/127.5 - 1 의 범위는 -1 ~ 1 # 정규화라기 보다 표준화와 유사  Z-정규화
"""


### 확인부분 ###
# bbb = iter(train_loader)
# aaa = next(bbb)
# print(aaa[0].shape) #torch.Size([32, 1, 56, 56]) (배치, 채널, 가로, 세로)
# print(len(train_loader)) # 1875 = 6000 / 32

#2. model
class CNN(nn.Module):
    def __init__(self, num_featueres):
        super(CNN, self, ).__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_featueres, 64, kernel_size=(3, 3), stride=1),  # n, 64, 54, 54
            # model.add(Conv2d(64, (3, 3), stride = 1, input_shape = (56, 56, 1)))
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),   # n, 64, 27, 27
            nn.Dropout(0.2)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1), #(n, 32, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), #(n, 32, 12, 12)
            nn.Dropout(0.2)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1), #(n, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)), #(n, 16, 5, 5)
            nn.Dropout(0.2)
        )
        self.hidden_layer4 = nn.Linear(16*5*5, 16)
        self.output = nn.Linear(in_features=16, out_features=10)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = x.view(x.shape[0], -1) # flatten() layer와 동일한 역할
        x = self.hidden_layer4(x)
        x = self.output(x)

        return x

model = CNN(1).to(DEVICE)

# model.summary()
print(model)

"""
CNN(
  (hidden_layer1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.2, inplace=False)
  )
  (hidden_layer2): Sequential(
    (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.2, inplace=False)
  )
  (hidden_layer3): Sequential(
    (0): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.2, inplace=False)
  )
  (hidden_layer4): Linear(in_features=400, out_features=16, bias=True)
  (output): Linear(in_features=16, out_features=10, bias=True)
)
"""
from torchsummary import summary
summary(model, (1, 56, 56))

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 64, 54, 54]             640
#               ReLU-2           [-1, 64, 54, 54]               0
#          MaxPool2d-3           [-1, 64, 27, 27]               0
#            Dropout-4           [-1, 64, 27, 27]               0
#             Conv2d-5           [-1, 32, 25, 25]          18,464
#               ReLU-6           [-1, 32, 25, 25]               0
#          MaxPool2d-7           [-1, 32, 12, 12]               0
#            Dropout-8           [-1, 32, 12, 12]               0
#             Conv2d-9           [-1, 16, 10, 10]           4,624
#              ReLU-10           [-1, 16, 10, 10]               0
#         MaxPool2d-11             [-1, 16, 5, 5]               0
#           Dropout-12             [-1, 16, 5, 5]               0
#            Linear-13                   [-1, 16]           6,416
#            Linear-14                   [-1, 10]             170
# ================================================================
# Total params: 30,314
# Trainable params: 30,314
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.01
# Forward/backward pass size (MB): 3.97
# Params size (MB): 0.12
# Estimated Total Size (MB): 4.09
# ----------------------------------------------------------------

"""
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

        epoch_loss += loss.item()
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
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
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(loader), epoch_acc / len(loader)
# loss, acc = model.evaluate(x_test, y_test)

EPOCH = 100
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    print(f'epoch : {epoch}, loss : {loss:.4f}, acc : {acc:.3f}, val_loss : {val_loss:.4f}, val_acc : {val_acc:.3f}')

#4. 평가, 예측
et = time.time()
loss, acc = evaluate(model, criterion, test_loader)
print("================================================================================")
print('걸린시간', et - st)
print('최종 Loss :', loss)
print('최종 acc :', acc)


걸린시간 1259.9318356513977
최종 Loss : 0.03699756304447252
최종 acc : 0.987020766773163
"""
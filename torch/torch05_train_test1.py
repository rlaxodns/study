import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print("torch", torch.__version__, "사용기기", DEVICE)

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10,11])
y_test = np.array([8,9,10,11])

x_predict = np.array([12,13,14])

# print(x_train.shape, y_train.shape) #(7,) (7,)

x_train =torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
y_train =torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test =torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_test =torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

# print(x_train.shape, y_train.shape) #torch.Size([7, 1]) torch.Size([7, 1])
x_predict =torch.FloatTensor(x_predict).unsqueeze(1).to(DEVICE)
print(x_predict.shape)


#2. model
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.Linear(64, 64),
    nn.Linear(64, 64),
    nn.Linear(64, 1)
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
    loss = train(model, criterion, optimizer, x_train, y_train)
    print("epoch", epoch, "loss", loss)

#4. 평가 및 예측
def evaluate(model, criterion, x, y):
    model.eval()

    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)

    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print(loss2)

result = model(x_predict.to(DEVICE))
print(result.detach().cpu().numpy())

"""
0.0006817805697210133
[[12.033417 ]
 [13.0364275]
 [14.039435 ]]
 """
import matplotlib.pyplot as plt

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

# 학습용 데이터와 평가용 데이터 분리
training_data = MNIST(root="./", train=True, download=True, transform=ToTensor())
test_data = MNIST(root="./", train=False, download=True, transform=ToTensor())

print(len(training_data)) # 학습에 사용할 데이터 개수
print(len(test_data)) # 평가에 사용할 데이터 개수

for i in range(9): # 샘플 이미지 9개 출력
    plt.subplot(3, 3,i+1)
    plt.imshow(training_data.data[i])
plt.show()

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)

# 평가용은 데이터를 섞을 필요가 없음
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 학습에 사용할 프로세서 지정
device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
model.to(device) # 모델의 파라미터를 GPU로 보냄

lr = 1e-3
optim = Adam(model.parameters(), lr=lr)

for epoch in range(20):
    for data, label in train_loader:
        optim.zero_grad()
        # 입력 데이터 모양을 모델의 입력에 맞게 변환
        data = torch.reshape(data, (-1, 784)).to(device)
        preds = model(data)

        loss = nn.CrossEntropyLoss()(preds, label.to(device)) # 손실 계산
        loss.backward() # 오차 역전파

        optim.step() # 최적화 진행
    
    print(f"epoch{epoch+1} loss:{loss.item()}")

# 모델을 MNIST.pth라는 이름으로 저장
torch.save(model.state_dict(), "MNIST.pth")

# 모델 가중치 불러오기
model.load_state_dict(torch.load("MNIST.pth", map_location=device))

num_corr = 0 # 분류에 성공한 전체 개수

with torch.no_grad(): # 기울기를 계산하지 않음
    for data, label in test_loader:
        data = torch.reshape(data, (-1, 784)).to(device)

        output = model(data.to(device))
        preds = output.data.max(1)[1] # 모델의 예측값 계산

        # 올바르게 분류한 개수
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

    print(f"Accuracy:{num_corr/len(test_data)}") # 분류 정확도 출력
    
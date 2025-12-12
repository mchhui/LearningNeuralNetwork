# Epoch 1/50, Test Accuracy: 53.44%, LR: 0.001000
# Epoch 2/50, Test Accuracy: 59.97%, LR: 0.001000
# Epoch 3/50, Test Accuracy: 60.63%, LR: 0.001000
# Epoch 4/50, Test Accuracy: 65.13%, LR: 0.001000
# Epoch 5/50, Test Accuracy: 68.87%, LR: 0.001000
# Epoch 6/50, Test Accuracy: 68.33%, LR: 0.001000
# Epoch 7/50, Test Accuracy: 66.99%, LR: 0.001000
# Epoch 8/50, Test Accuracy: 72.34%, LR: 0.001000
# Epoch 9/50, Test Accuracy: 73.91%, LR: 0.001000
# Epoch 10/50, Test Accuracy: 75.01%, LR: 0.001000
# Epoch 11/50, Test Accuracy: 77.81%, LR: 0.001000
# Epoch 12/50, Test Accuracy: 77.51%, LR: 0.001000
# Epoch 13/50, Test Accuracy: 77.35%, LR: 0.001000
# Epoch 14/50, Test Accuracy: 76.16%, LR: 0.001000
# Epoch 15/50, Test Accuracy: 77.42%, LR: 0.000500
# Epoch 16/50, Test Accuracy: 81.02%, LR: 0.000500
# Epoch 17/50, Test Accuracy: 80.86%, LR: 0.000500
# Epoch 18/50, Test Accuracy: 79.22%, LR: 0.000500
# Epoch 19/50, Test Accuracy: 78.67%, LR: 0.000500
# Epoch 20/50, Test Accuracy: 80.50%, LR: 0.000500
# Epoch 21/50, Test Accuracy: 81.68%, LR: 0.000500
# Epoch 22/50, Test Accuracy: 79.53%, LR: 0.000500
# Epoch 23/50, Test Accuracy: 81.83%, LR: 0.000500
# Epoch 24/50, Test Accuracy: 79.08%, LR: 0.000500
# Epoch 25/50, Test Accuracy: 81.31%, LR: 0.000500
# Epoch 26/50, Test Accuracy: 80.94%, LR: 0.000500
# Epoch 27/50, Test Accuracy: 81.04%, LR: 0.000500
# Epoch 28/50, Test Accuracy: 80.46%, LR: 0.000500
# Epoch 29/50, Test Accuracy: 81.05%, LR: 0.000500
# Epoch 30/50, Test Accuracy: 82.00%, LR: 0.000250
# Epoch 31/50, Test Accuracy: 83.72%, LR: 0.000250
# Epoch 32/50, Test Accuracy: 82.19%, LR: 0.000250
# Epoch 33/50, Test Accuracy: 82.34%, LR: 0.000250
# Epoch 34/50, Test Accuracy: 81.99%, LR: 0.000250
# Epoch 35/50, Test Accuracy: 82.33%, LR: 0.000250
# Epoch 36/50, Test Accuracy: 83.21%, LR: 0.000250
# Epoch 37/50, Test Accuracy: 83.77%, LR: 0.000250
# Epoch 38/50, Test Accuracy: 83.08%, LR: 0.000250
# Epoch 39/50, Test Accuracy: 82.39%, LR: 0.000250
# Epoch 40/50, Test Accuracy: 81.99%, LR: 0.000250
# Epoch 41/50, Test Accuracy: 83.34%, LR: 0.000250
# Epoch 42/50, Test Accuracy: 82.13%, LR: 0.000250
# Epoch 43/50, Test Accuracy: 83.59%, LR: 0.000250
# Epoch 44/50, Test Accuracy: 82.97%, LR: 0.000250
# Epoch 45/50, Test Accuracy: 84.03%, LR: 0.000125
# Epoch 46/50, Test Accuracy: 83.63%, LR: 0.000125
# Epoch 47/50, Test Accuracy: 83.51%, LR: 0.000125
# Epoch 48/50, Test Accuracy: 84.29%, LR: 0.000125
# Epoch 49/50, Test Accuracy: 83.73%, LR: 0.000125
# Epoch 50/50, Test Accuracy: 82.16%, LR: 0.000125

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

imgWidth = 32
imgHeight = 32
# 输入层 32x32x3 width height channel(r,g,b)
# mlpconv_1
# -conv 192个5*5 step:1 padding:2
# -conv 160个1*1 step：1 padding:0
# -batchnorm
# -dropout
# -relu
# pool 2*2 step：2 max
#
# mlpconv_2
# -conv 192个5*5 step：1 padding：2
# -conv 192个1*1 step：1 padding：0
# -batchnorm
# -dropout
# -relu
# pool 3*3 step：2 average
#
# mlpconv_3
# -conv 192个3*3 step：1 padding：1
# -conv 192个1*1 step：1 padding：0
# -batchnorm
# -dropout
# -relu
# -conv 10个1*1 step：1 padding：0
# GAP
# softmax
class GAP_CNN(nn.Module):
    def __init__(self):
        super(GAP_CNN, self).__init__()
        # mlpconv_1
        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.conv1_1 = nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(160)
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # mlpconv_2
        self.conv2 = nn.Conv2d(160, 192, kernel_size=5, stride=1, padding=2)
        self.conv2_1 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(192)
        self.dropout2 = nn.Dropout(0.3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)

        # mlpconv_3
        self.conv3 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(192)
        self.dropout3 = nn.Dropout(0.3)
        self.relu3 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

    def forward(self, x):
        # mlpconv_1
        x = self.conv1(x)
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # mlpconv_2
        x = self.conv2(x)
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # mlpconv_3
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.relu3(x)
        x = self.conv3_2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, 10)

        return x

model = GAP_CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)  # 学习率衰减

batch_size=64
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 数据增强
    transforms.RandomHorizontalFlip(),     # 数据增强
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练函数
def train():
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

epochs = 50  # 增加训练轮数
for epoch in range(epochs):
    train()
    accuracy = test()
    scheduler.step()  # 更新学习率
    print(f'Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')

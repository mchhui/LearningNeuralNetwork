# Epoch 1/20, Test Accuracy: 49.52%
# Epoch 2/20, Test Accuracy: 62.62%
# Epoch 3/20, Test Accuracy: 66.20%
# Epoch 4/20, Test Accuracy: 67.66%
# Epoch 5/20, Test Accuracy: 69.30%
# Epoch 6/20, Test Accuracy: 68.66%
# Epoch 7/20, Test Accuracy: 68.50%
# Epoch 8/20, Test Accuracy: 67.70%
# Epoch 9/20, Test Accuracy: 67.78%
# Epoch 10/20, Test Accuracy: 65.81%
# Epoch 11/20, Test Accuracy: 66.79%
# Epoch 12/20, Test Accuracy: 66.62%
# Epoch 13/20, Test Accuracy: 67.45%
# Epoch 14/20, Test Accuracy: 66.72%
# Epoch 15/20, Test Accuracy: 66.88%
# Epoch 16/20, Test Accuracy: 67.19%
# Epoch 17/20, Test Accuracy: 66.60%
# Epoch 18/20, Test Accuracy: 67.18%
# Epoch 19/20, Test Accuracy: 67.03%
# Epoch 20/20, Test Accuracy: 66.69%

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

imgWidth = 64
imgHeight = 64
# Conv2d和MaxPool2d都默认0维是批次 ReLU不在乎维度 Linear只要求最后一个维度是特征 前面的都视为批次维度
# 输入层 64x64x3 width height channel(r,g,b)
# 卷积层 32个3x3x3 2D卷积核
# 激活函数 ReLU
# 池化层 2x2 最大池化
# 卷积层 32个3x3x3 2D卷积核
# 激活函数 ReLU
# 池化层 2x2 最大池化
# 全连接层 256 个神经元
# 激活函数 ReLU
# 全连接层 128 个神经元
# 激活函数 ReLU
# 全连接层 64 个神经元
# 激活函数 ReLU
# 输出层 10 个神经元 softmax激活函数
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        #LeNet-5 未使用 padding
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算卷积层输出尺寸: 64->62->31->29->14
        size = 14 * 14 * 32
        self.fc1 = nn.Linear(size, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.out = nn.Linear(64, 10)
        # 注意：不在模型中使用Softmax，因为交叉熵损失函数会自动处理
        # 如果需要概率输出，可以在推理时单独应用Softmax

    def forward(self, x):
        # 卷积层1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 卷积层2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.relu5(x)
        x = self.out(x)

        # 返回logits（未经Softmax处理的原始输出）
        return x

model = BasicCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size=64
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 上采样到64x64
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

epochs = 5
for epoch in range(epochs):
    train()
    accuracy = test()
    print(f'Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%')

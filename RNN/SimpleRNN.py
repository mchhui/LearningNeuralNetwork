# Using device: cuda
# 开始训练...
# Epoch 1/20, Loss: 0.6990
# Epoch 2/20, Loss: 0.6889
# Epoch 3/20, Loss: 0.6717
# Epoch 4/20, Loss: 0.6463
# Epoch 5/20, Loss: 0.6099
# Epoch 6/20, Loss: 0.5712
# Epoch 7/20, Loss: 0.5400
# Epoch 8/20, Loss: 0.5036
# Epoch 9/20, Loss: 0.4916
# Epoch 10/20, Loss: 0.4754
# Epoch 11/20, Loss: 0.4633
# Epoch 12/20, Loss: 0.4445
# Epoch 13/20, Loss: 0.4335
# Epoch 14/20, Loss: 0.4202
# Epoch 15/20, Loss: 0.4242
# Epoch 16/20, Loss: 0.4212
# Epoch 17/20, Loss: 0.4420
# Epoch 18/20, Loss: 0.4285
# Epoch 19/20, Loss: 0.4099
# Epoch 20/20, Loss: 0.4295
# 训练完成，开始测试...
# Test Accuracy: 0.5424

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Embedding vocab_size=10002, output_size=128
# RNN input_size=128, hidden_size=128
# Linear output_size=1
# sigmoid (for binary classification)
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size=10002, embedding_dim=128, hidden_size=128):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, hn = self.rnn(x)
        out = self.linear(hn.squeeze(0))
        return self.sigmoid(out)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
data = torch.load('preprocessed_imdb.pt')
train_texts = data['train_texts'].to(device)
train_labels = data['train_labels'].float().unsqueeze(1).to(device)
test_texts = data['test_texts'].to(device)
test_labels = data['test_labels'].float().unsqueeze(1).to(device)

train_dataset = TensorDataset(train_texts, train_labels)
test_dataset = TensorDataset(test_texts, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, optimizer
model = SimpleRNN().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
def train_model(num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

# Testing
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

print("开始训练...")
train_model()
print("训练完成，开始测试...")
test_model()

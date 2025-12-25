# Using device: cuda
# 开始训练...
# Epoch 1/20, Loss: 0.4857
# Epoch 2/20, Loss: 0.3130
# Epoch 3/20, Loss: 0.2389
# Epoch 4/20, Loss: 0.1774
# Epoch 5/20, Loss: 0.1334
# Epoch 6/20, Loss: 0.0989
# Epoch 7/20, Loss: 0.0726
# Epoch 8/20, Loss: 0.0562
# Epoch 9/20, Loss: 0.0486
# Epoch 10/20, Loss: 0.0377
# Epoch 11/20, Loss: 0.0381
# Epoch 12/20, Loss: 0.0357
# Epoch 13/20, Loss: 0.0323
# Epoch 14/20, Loss: 0.0273
# Epoch 15/20, Loss: 0.0275
# Epoch 16/20, Loss: 0.0241
# Epoch 17/20, Loss: 0.0198
# Epoch 18/20, Loss: 0.0272
# Epoch 19/20, Loss: 0.0229
# Epoch 20/20, Loss: 0.0194
# 训练完成，开始测试...
# Test Accuracy: 0.8515

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query.unsqueeze(1)) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class RNNsearch(nn.Module):
    def __init__(self, vocab_size=10002, embedding_dim=128, hidden_size=128):
        super(RNNsearch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.RNN(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.encoder_output_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.attention = BahdanauAttention(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        encoder_outputs, encoder_hidden = self.encoder(embedded)

        # 双向RNN的隐藏状态: (num_layers*2, batch_size, hidden_size)
        # 前向: encoder_hidden[0], 后向: encoder_hidden[1]
        encoder_hidden_fwd = encoder_hidden[0]  # 正向隐藏状态
        encoder_hidden_bwd = encoder_hidden[1]  # 反向隐藏状态

        # 混合正向和反向隐藏状态
        combined_hidden = torch.cat([encoder_hidden_fwd, encoder_hidden_bwd], dim=-1)
        decoder_hidden = self.encoder_output_proj(combined_hidden).unsqueeze(0)

        # 对encoder_outputs也进行同样的投影
        encoder_outputs = self.encoder_output_proj(encoder_outputs)

        context, _ = self.attention(decoder_hidden.squeeze(0), encoder_outputs)

        decoder_output, _ = self.decoder(context, decoder_hidden)
        decoder_output = decoder_output.squeeze(1)

        out = self.linear(decoder_output)
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
model = RNNsearch().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
def train_model(num_epochs=50):
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

# Using device: cuda
# 开始训练...
# Epoch 1/50, Loss: 0.6711, LR: 0.000100
# Epoch 2/50, Loss: 0.5366, LR: 0.000100
# Epoch 3/50, Loss: 0.4775, LR: 0.000100
# Epoch 4/50, Loss: 0.4356, LR: 0.000100
# Epoch 5/50, Loss: 0.3992, LR: 0.000100
# Epoch 6/50, Loss: 0.3758, LR: 0.000100
# Epoch 7/50, Loss: 0.3531, LR: 0.000100
# Epoch 8/50, Loss: 0.3342, LR: 0.000100
# Epoch 9/50, Loss: 0.3169, LR: 0.000100
# Epoch 10/50, Loss: 0.2959, LR: 0.000100
# Epoch 11/50, Loss: 0.2769, LR: 0.000100
# Epoch 12/50, Loss: 0.2593, LR: 0.000100
# Epoch 13/50, Loss: 0.2415, LR: 0.000100
# Epoch 14/50, Loss: 0.2263, LR: 0.000100
# Epoch 15/50, Loss: 0.2175, LR: 0.000100
# Epoch 16/50, Loss: 0.2064, LR: 0.000100
# Epoch 17/50, Loss: 0.1872, LR: 0.000100
# Epoch 18/50, Loss: 0.1750, LR: 0.000100
# Epoch 19/50, Loss: 0.1600, LR: 0.000100
# Epoch 20/50, Loss: 0.1472, LR: 0.000100
# Epoch 21/50, Loss: 0.1449, LR: 0.000100
# Epoch 22/50, Loss: 0.1253, LR: 0.000100
# Epoch 23/50, Loss: 0.1136, LR: 0.000100
# Epoch 24/50, Loss: 0.1131, LR: 0.000100
# Epoch 25/50, Loss: 0.1018, LR: 0.000050
# Epoch 26/50, Loss: 0.0816, LR: 0.000050
# Epoch 27/50, Loss: 0.0731, LR: 0.000050
# Epoch 28/50, Loss: 0.0710, LR: 0.000050
# Epoch 29/50, Loss: 0.0702, LR: 0.000050
# Epoch 30/50, Loss: 0.0648, LR: 0.000050
# Epoch 31/50, Loss: 0.0572, LR: 0.000050
# Epoch 32/50, Loss: 0.0593, LR: 0.000050
# Epoch 33/50, Loss: 0.0603, LR: 0.000050
# Epoch 34/50, Loss: 0.0523, LR: 0.000050
# Epoch 35/50, Loss: 0.0512, LR: 0.000050
# Epoch 36/50, Loss: 0.0517, LR: 0.000050
# Epoch 37/50, Loss: 0.0508, LR: 0.000050
# Epoch 38/50, Loss: 0.0497, LR: 0.000050
# Epoch 39/50, Loss: 0.0529, LR: 0.000050
# Epoch 40/50, Loss: 0.0465, LR: 0.000050
# Epoch 41/50, Loss: 0.0486, LR: 0.000050
# Epoch 42/50, Loss: 0.0480, LR: 0.000050
# Epoch 43/50, Loss: 0.0506, LR: 0.000050
# Epoch 44/50, Loss: 0.0497, LR: 0.000050
# Epoch 45/50, Loss: 0.0455, LR: 0.000050
# Epoch 46/50, Loss: 0.0434, LR: 0.000050
# Epoch 47/50, Loss: 0.0460, LR: 0.000050
# Epoch 48/50, Loss: 0.0420, LR: 0.000050
# Epoch 49/50, Loss: 0.0422, LR: 0.000050
# Epoch 50/50, Loss: 0.0395, LR: 0.000025
# 训练完成，开始测试...
# Test Accuracy: 0.8063

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, Q, K, V, src_mask=None):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if src_mask is not None:
            attn_mask = src_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.num_heads, Q.size(2), -1)
        else:
            attn_mask = None

        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, attn_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        attn_output = self.dropout(attn_output)

        output = self.W_o(attn_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AddAndNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))


class Encoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_and_norm1 = AddAndNorm(d_model, dropout)
        self.add_and_norm2 = AddAndNorm(d_model, dropout)


    def forward(self, x, src_mask=None):
        attn_output = self.multi_head_attention(x, x, x, src_mask)
        x = self.add_and_norm1(x, attn_output)

        ff_output = self.feed_forward(x)
        x = self.add_and_norm2(x, ff_output)

        return x

class EncoderNN(nn.Module):
    def __init__(self, vocab_size=10002, d_model=256, num_heads=8, d_ff=1024,
                 num_layers=2, dropout=0.1, max_len=200):
        super(EncoderNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoders = nn.ModuleList([
            Encoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)  # 二分类输出

    def forward(self, x):
        src_mask = (x != 0).float()

        x = self.embedding(x)
        x = self.positional_encoding(x)

        for encoder in self.encoders:
            x = encoder(x, src_mask)

        cls_token = x[:, 0, :]

        x = self.dropout(cls_token)
        output = self.classifier(x)
        return output  # 返回logits，形状为 [batch_size, 1]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data = torch.load('preprocessed_imdb.pt')
train_texts = data['train_texts'].to(device)
train_labels = data['train_labels'].float().unsqueeze(1).to(device)
test_texts = data['test_texts'].to(device)
test_labels = data['test_labels'].float().unsqueeze(1).to(device)

train_dataset = TensorDataset(train_texts, train_labels)
test_dataset = TensorDataset(test_texts, test_labels)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = EncoderNN().to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

def train_model(num_epochs=200):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, LR: {current_lr:.6f}')

def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predicted = (torch.sigmoid(outputs.squeeze(-1)) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.squeeze(-1)).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

print("开始训练...")
train_model()
print("训练完成，开始测试...")
test_model()

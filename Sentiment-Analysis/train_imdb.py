import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np

DATA_DIR = "aclImdb/train"

############################################
# 1) 读取数据
############################################
def load_imdb(data_dir):
    texts, labels = [], []
    for label in ["pos", "neg"]:
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

train_texts, train_labels = load_imdb(DATA_DIR)
print("数据加载完成，共:", len(train_texts))

############################################
# 2) 简单文本清洗 + 分词
############################################
def tokenize(text):
    return text.lower().replace("\n"," ").split()

train_texts = [tokenize(t) for t in train_texts]

############################################
# 3) 构建词表
############################################
word2idx = {"<PAD>":0, "<UNK>":1}
for text in train_texts:
    for w in text:
        if w not in word2idx:
            word2idx[w] = len(word2idx)

print("Vocabulary Size:", len(word2idx))

def text_to_ids(tokens, max_len=200):
    ids = [word2idx.get(w,1) for w in tokens]
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return ids

X = [text_to_ids(t) for t in train_texts]

############################################
# 4) 标签编码
############################################
le = LabelEncoder()
y = le.fit_transform(train_labels)

train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1)
train_x, val_x = torch.tensor(train_x), torch.tensor(val_x)
train_y, val_y = torch.tensor(train_y), torch.tensor(val_y)

############################################
# 5) DataLoader
############################################
class IMDBDataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(IMDBDataset(train_x, train_y), batch_size=64, shuffle=True)
val_loader = DataLoader(IMDBDataset(val_x, val_y), batch_size=64)

############################################
# 6) 模型定义：Embedding + LSTM + 分类层
############################################
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden*2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)  # 双向拼接
        return self.fc(h)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentimentModel(len(word2idx)).to(device)

############################################
# 7) 训练循环
############################################
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("开始训练...")
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    print(f"Epoch:{epoch+1}, Loss:{total_loss:.2f}, Val Acc:{correct/total:.4f}")

# 保存模型
torch.save(model.state_dict(), "imdb_lstm.pth")
print("训练完成，模型已保存 → imdb_lstm.pth")

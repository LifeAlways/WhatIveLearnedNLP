import torch
from train_imdb import SentimentModel, word2idx, text_to_ids, le

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentimentModel(len(word2idx))
model.load_state_dict(torch.load("imdb_lstm.pth"))
model.to(device).eval()

text = "This movie was surprisingly good and touching."
ids = torch.tensor([text_to_ids(text.lower().split())]).to(device)

pred = model(ids).argmax(1).item()
print("预测情感:", le.inverse_transform([pred])[0])

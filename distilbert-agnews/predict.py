# predict.py
import sys
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

MODEL_DIR = "./models/distilbert-agnews"

label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model

def predict(text):
    tokenizer, model = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()
    return label_map[pred]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python predict.py \"Your text here\"")
    else:
        text = sys.argv[1]
        print("输入文本:", text)
        print("预测分类:", predict(text))

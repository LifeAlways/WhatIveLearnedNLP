# train_distilbert.py
import os
import torch
from datasets import load_from_disk,load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

DATA_DIR = "./data"  # 和 download_data.py 保持一致

print("正在加载本地数据...")
dataset = load_dataset("parquet", data_files={
    "train": os.path.join(DATA_DIR, "train-00000-of-00001.parquet"),
    "test": os.path.join(DATA_DIR, "test-00000-of-00001.parquet")
})  #网络不行改为手动下载
train_data = dataset["train"]
test_data = dataset["test"]

print("正在加载模型 ...")
tokenizer = DistilBertTokenizerFast.from_pretrained("./models/distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "./models/distilbert-base-uncased",
    num_labels=4,
    ignore_mismatched_sizes=True
)

model.to(device)


def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_tokenized = train_data.map(tokenize_fn, batched=True)
test_tokenized = test_data.map(tokenize_fn, batched=True)



training_args = TrainingArguments(
    output_dir="./models/distilbert-agnews",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    logging_steps=100,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
)

print("开始训练 ...")
trainer.train()

model.save_pretrained("./models/distilbert-agnews")
tokenizer.save_pretrained("./models/distilbert-agnews")

print("模型训练完成，已保存到 ./models/distilbert-agnews")

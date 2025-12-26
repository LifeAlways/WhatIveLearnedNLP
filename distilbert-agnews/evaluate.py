# python
# evaluate.py
import os
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = "./data"
MODEL_DIR = "./models/distilbert-agnews"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def evaluate_model():
    # 与训练脚本保持一致地从 parquet 文件加载测试集
    dataset = load_dataset("parquet", data_files={
        "test": os.path.join(DATA_DIR, "test-00000-of-00001.parquet")
    })
    test_data = dataset["test"]

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    test_tokenized = test_data.map(tokenize_fn, batched=True)

    # 保留 input_ids, attention_mask 和 label（如果存在），并设置为 torch 格式以供 Trainer.predict 使用
    cols = ["input_ids", "attention_mask"]
    if "label" in test_tokenized.column_names:
        cols.append("label")
    test_tokenized.set_format(type="torch", columns=cols)

    trainer = Trainer(model=model, tokenizer=tokenizer)
    preds = trainer.predict(test_tokenized)

    accuracy = compute_metrics((preds.predictions, preds.label_ids))
    print("测试集准确率:", accuracy)

    # 详细报告
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)
    print("\n分类报告:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    evaluate_model()


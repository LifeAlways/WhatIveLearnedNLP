# download_model.py
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# 设置代理（根据需要替换代理地址）
proxy = "http://127.0.0.1:7890"  # 替换为你的代理地址
os.environ["HTTPS_PROXY"] = proxy
os.environ["HTTP_PROXY"] = proxy

# 定义模型保存目录
MODEL_DIR = "./models/distilbert-base-uncased"


def download_model():
    # 使用 Hugging Face 下载模型和 tokenizer
    model_name = "distilbert/distilbert-base-uncased"
    print("正在下载模型和tokenizer...")

    # 下载模型和tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)

    # 将模型和tokenizer保存到本地
    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

    print(f"模型已保存到 {MODEL_DIR}")


if __name__ == "__main__":
    download_model()

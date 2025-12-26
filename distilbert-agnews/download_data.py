# download_data.py
from datasets import load_dataset
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

SAVE_PATH = "./data"  # 本地数据保存位置
os.makedirs(SAVE_PATH, exist_ok=True)

print("正在下载 AG News 数据...")
dataset = load_dataset("ag_news")

print("正在保存到本地 ...")
dataset.save_to_disk(SAVE_PATH)

print(f"下载完成，数据已保存在: {SAVE_PATH}")

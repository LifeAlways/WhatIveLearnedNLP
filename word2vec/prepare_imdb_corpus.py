import os

IMDB_DIR = "../Sentiment-Analysis/aclImdb"

# 输出路径
OUTPUT_FILE = "data/imdb.txt"

def read_folder(path):
    texts = []
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read().replace("\n", " "))
    return texts

def build_corpus():
    parts = ["train/pos", "train/neg", "test/pos", "test/neg"]
    all_texts = []

    for p in parts:
        folder_path = os.path.join(IMDB_DIR, p)
        print(f"正在处理: {folder_path}")
        all_texts.extend(read_folder(folder_path))

    print(f"数据量: {len(all_texts)} 条影评")

    # 输出到 data/imdb.txt
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in all_texts:
            f.write(line + "\n")

    print(f"处理完成！语料已保存到 → {OUTPUT_FILE}")

if __name__ == "__main__":
    build_corpus()

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# ============ 1. 读取文本数据 ============
def load_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        corpus = [simple_preprocess(line) for line in f]
    return corpus

corpus = load_corpus("data/imdb.txt")  # 你的训练语料

# ============ 2. 训练Word2Vec模型 ============
model = Word2Vec(
    sentences=corpus,
    vector_size=100,  # 词向量维度
    window=5,         # 上下文窗口
    min_count=5,      # 词最少出现次数
    workers=4,        # CPU核心数量
    sg=1              # 1=SkipGram, 0=CBOW
)

# ============ 3. 保存模型 ============
model.save("models/word2vec.model")
model.wv.save_word2vec_format("models/word2vec_vectors.txt")

print("模型训练完成！已保存。")

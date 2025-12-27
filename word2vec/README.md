# Word2Vec on IMDB Reviews

这是一个基于 IMDB 电影评论数据集训练 Word2Vec 词向量模型的实践项目。通过该项目深入理解词嵌入的原理、训练过程以及语义关系的捕捉能力。

## 项目动机

在 NLP 学习过程中，词向量是理解现代自然语言处理的基础。虽然 BERT、GPT 等预训练模型已经成为主流，但作为一名初学者，从零开始训练 Word2Vec 能够帮助我：

- **理解词嵌入的本质**：词向量如何通过共现关系捕捉语义
- **掌握超参数调优**：window size、vector_size、min_count 等参数的实际影响
- **观察领域特性**：在情感分析场景下，词向量如何反映情感极性

## 项目结构

```
.
├── prepare_imdb_corpus.py   # 数据预处理脚本
├── train_word2vec.py         # 模型训练脚本
├── result_show.py            # 结果展示脚本
├── requirements.txt          # 项目依赖
├── data/
│   └── imdb.txt             # 处理后的语料
└── models/
    ├── word2vec.model       # 完整模型
    └── word2vec_vectors.txt # 词向量文件
```

## 核心思考

### 1. 数据处理策略

在 `prepare_imdb_corpus.py` 中，我选择了简单但有效的处理方式：
- 合并 train/test、pos/neg 四个子集，最大化训练数据量（50,000 条评论）
- 保留原始文本结构，仅去除换行符，让模型学习自然的句子边界
- 没有进行过度的文本清洗，保留了真实的用户表达方式


### 2. 模型参数选择

```python
vector_size=100  # 词向量维度
window=5         # 上下文窗口
min_count=5      # 最小词频
sg=1             # Skip-gram 模式
```

**为什么选择 Skip-gram？**
- Skip-gram 在小数据集上表现更好
- 对低频词更友好，而影评中很多形容词频率适中但语义重要
- 能够学习到更细粒度的语义关系

**窗口大小的权衡**：
- window=5 是一个经典设置，能够捕捉短语级别的语义
- 更大的窗口会捕捉主题相关性，更小的窗口捕捉句法相关性
- 对于情感分析，5 是一个平衡点

### 3. 实验结果分析

运行 `result_show.py` 得到的结果揭示了模型的学习质量：

#### 相似度对比
```
similarity("good", "great") = 0.79
similarity("good", "bad")   = 0.77
```
模型确实捕捉到了 good-great 的正向关联，但也体现了需要更复杂的模型来区分极性。

#### 电影领域词汇
```
"film" → movie (0.92), flick (0.73), picture (0.72)
```
完美捕捉了同义替换关系，证明模型学到了领域知识。

## 快速开始

### 环境配置
```bash
pip install -r requirements.txt
```

### 数据准备
下载 [IMDB 数据集](http://ai.stanford.edu/~amaas/data/sentiment/)，解压后调整 `prepare_imdb_corpus.py` 中的路径：
```python
IMDB_DIR = "你的路径/aclImdb"
```

运行数据预处理：
```bash
python prepare_imdb_corpus.py
```

### 训练模型
```bash
python train_word2vec.py
```

### 查看结果
```bash
python result_show.py
```


## 学习总结

这个项目让我深刻体会到：
- **词向量不一定能展现正反向关系**：它们只是统计共现关系的压缩表示
- **领域数据的重要性**：在 IMDB 数据上训练的模型，天然理解电影评论的语言风格
- **评估的多维性**：相似度高不等于语义正确，需要结合任务具体分析

词嵌入是 NLP 的起点，但理解它的能力边界，才能更好地使用 BERT、GPT 等更强大的模型。

## 依赖库

- `gensim` >= 4.0：Word2Vec 训练
- `numpy` >= 1.20：数值计算
- `tqdm` >= 4.60：进度显示

## 参考资料

- [Word2Vec 原论文](https://arxiv.org/abs/1301.3781) - Mikolov et al., 2013
- [Gensim 官方文档](https://radimrehurek.com/gensim/)
- [IMDB 数据集](http://ai.stanford.edu/~amaas/data/sentiment/)

## 项目完成时间
- 2025.11.23

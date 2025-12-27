# 自然语言处理课程学习项目集

> 2025-2026 学年第一学期 NLP 课程自主学习成果

## 基本信息

| 项目     | 内容                |
|--------|-------------------|
| **姓名** | `马祺喆`             |
| **学号** | `42342009`        |
| **专业** | `人工智能（智能金融光华实验班）` |
| **课程** | 自然语言处理            |
| **学期** | 2025-2026-1       |

---

## 项目概览

本仓库记录了我本学期在自然语言处理课程中的自主学习实践，涵盖了从词向量到预训练模型的完整学习路径。

| 序号 | 项目名称 | 核心技术 | 完成时间 |
|:----:|----------|----------|:--------:|
| 1 | IMDB 情感分析 (LSTM) | PyTorch + LSTM + Embedding | 2025.10.20 |
| 2 | Word2Vec 词向量训练 | Gensim + Skip-gram | 2025.11.23 |
| 3 | AG News 文本分类 | DistilBERT + Fine-tuning | 2025.12.01 |
| 4 | SQuAD 问答系统 | MobileBERT + QA | 2025.12.20 |

---

## 项目详情



### 1. IMDB 情感分析 (LSTM)

**完成日期**：2025.10.20
**目录**: `Sentiment-Analysis/`

我参考 Word2Vec 原论文 (Mikolov et al., 2013)，理解了 Skip-gram 模型的训练过程和词向量的语义捕捉能力，其基本思想是通过预测中心词周围的上下文词来学习词的分布式表示，使得语义相近的词在向量空间中距离更近

基于 PyTorch 实现的电影评论情感二分类系统，使用双向 LSTM 捕捉文本的上下文语义。

- **数据集**: IMDB 电影评论（25,000 条训练数据）
- **模型架构**: Embedding → Bidirectional LSTM → FC
- **验证准确率**: 79.12%
- **学习重点**: 文本预处理、序列建模、训练循环实现

### 2. Word2Vec 词向量训练

**完成日期**：2025.11.23
**目录**: `word2vec/`

我参考 Word2Vec 原论文 (Mikolov et al., 2013)，理解了 Skip-gram 模型的训练过程和词向量的语义捕捉能力，其基本思想是通过预测中心词周围的上下文词来学习词的分布式表示，使得语义相近的词在向量空间中距离更近

从零开始在 IMDB 语料上训练 Word2Vec 模型，深入理解词嵌入的原理。

- **数据集**: IMDB 全量评论（50,000 条）
- **模型配置**: Skip-gram, 100维向量, window=5
- **核心发现**: 词向量能捕捉同义关系（film-movie: 0.92），但难以区分情感极性
- **学习重点**: 词嵌入原理、超参数调优、语义关系分析


### 3. AG News 文本分类

**完成日期**：2025.12.01
**目录**: `distilbert-agnews/`

我参考 DistilBERT 论文 (Sanh et al., 2019)，理解了知识蒸馏在模型压缩中的应用，其基本思想是让小模型学习大模型的软标签输出分布，从而在大幅减少参数量的同时保留大部分性能

使用 DistilBERT 预训练模型进行新闻四分类任务，体验迁移学习的威力。

- **数据集**: AG News（120,000 条新闻）
- **分类类别**: World / Sports / Business / Sci/Tech
- **测试准确率**: 93.97%
- **学习重点**: 预训练模型微调、Hugging Face 生态、分类评估指标

### 4. MobileBERT 问答系统

**完成日期**：2025.12.20
**目录**: `MobileBERT-SQuAD1_1/`

我参考 SQuAD: The Stanford Question Answering Dataset，理解了抽取式问答任务的建模方式和评估指标，其基本思想是将问答建模为在给定段落中预测答案起始和结束位置的 span extraction 任务，通过两个分类头分别预测 start 和 end 位置

在 SQuAD 1.1 数据集上微调 MobileBERT，实现抽取式问答系统。

- **数据集**: SQuAD 1.1（87,000 训练样本）
- **评估指标**: EM=73.34%, F1=83.00%
- **工程优化**: 梯度累积、显存优化、离线训练支持
- **学习重点**: 问答任务建模、答案位置映射、资源受限下的训练策略

---

## 学习路径与思考

```
词嵌入基础          序列建模           预训练模型          复杂任务
    │                  │                  │                  │
    ▼                  ▼                  ▼                  ▼
 Word2Vec    →    LSTM情感分析   →   DistilBERT分类   →   MobileBERT QA
(理解词向量)     (理解序列处理)      (理解迁移学习)       (理解端到端系统)
```

**核心收获**:

1. **从浅到深**: 从词向量的统计特性，到深度模型的表示学习
2. **从小到大**: 从自己训练的简单模型，到利用预训练模型的强大能力
3. **工程实践**: 数据处理、模型训练、评估分析的完整流程

各项目的具体学习内容请查看各项目内的README文件

---

## 环境配置

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install transformers datasets
pip install gensim numpy tqdm scikit-learn
```

各项目的详细依赖见对应目录下的 `requirements.txt`。

---

## 仓库结构

```
.
├── README.md                    # 本文件
├── Sentiment-Analysis/         # 项目1: LSTM情感分析
│   ├── aclImdb/
│   ├── imdb_lstm.pth
│   ├── train_imdb.py
│   ├── predict.py
│   ├── requirements.txt
│   ├── results.txt
│   └── README.md
├── word2vec/               # 项目2: Word2Vec训练
│   ├── data/
│   ├── models/
│   ├── requirements.txt
│   ├── results.txt
│   ├── prepare_imdb_corpus.py
│   ├── train_word2vec.py
│   ├── result_show.py
│   └── README.md
├── distilbert-agnews/           # 项目3: DistilBERT分类
│   ├── data/
│   ├── models
│   ├── tmp_trainer
│   ├── download_data.py
│   ├── download_model.py
│   ├── evaluate_results.txt
│   ├── requirements.txt
│   ├── train_distilbert.py
│   ├── evaluate.py
│   ├── predict.py
│   └── README.md
└── MobileBERT-SQuAD1_1/            # 项目4: MobileBERT问答
    ├── data/
    ├── models/
    ├── output/
    ├── requirements.txt
    ├── results.txt
    ├── download_data.py
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    └── README.md
```

---

## 参考资料

- [Word2Vec 原论文 (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [BERT 原论文 (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [DistilBERT 论文](https://arxiv.org/abs/1910.01108)
- [MobileBERT 论文](https://arxiv.org/abs/2004.02984)
- [Hugging Face 文档](https://huggingface.co/docs)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)

---


*最后更新: 2025年12月27日*
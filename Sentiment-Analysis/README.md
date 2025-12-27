# IMDB 情感分析项目

> 一个基于 PyTorch 的电影评论情感分类系统

## 项目概述

这是我在学习自然语言处理过程中实现的一个端到端的情感分析项目。通过对 IMDB 电影评论数据集的训练，模型能够判断一条评论是正面还是负面情感。

**核心技术栈：** PyTorch + LSTM + Embedding

## 设计思路与个人思考

### 选择架构的原因？

在设计模型时，我主要考虑了以下几点：

1. **嵌入层的必要性**
   - 文本数据无法直接输入神经网络，需要转换为稠密向量表示
   - 我选择了 128 维的词嵌入，这个维度在计算效率和表达能力之间取得了平衡
   - 相比 one-hot 编码，embedding 能够捕捉词与词之间的语义关系

2. **为什么使用 Bidirectional LSTM？**
   - 单向 LSTM 只能从前往后理解句子，但很多情感表达需要结合上下文
   - 例如："这部电影不算差" 和 "这部电影算不上好" 需要理解否定词的位置
   - 双向 LSTM 通过正向和反向两个方向的信息融合，更准确地捕捉情感

3. **序列长度设定为 200**
   - 分析了数据集后，我发现大部分评论在 200 个词以内
   - 过长的序列会增加计算成本，过短则可能丢失重要信息
   - 这是一个在实际应用中需要权衡的超参数

### 数据处理的考量

- **简单的分词策略：** 使用 `split()` 而非复杂的分词器，因为英文词间有明确空格分隔
- **词表大小达到 251,639：** 这是一个相对较大的词表，说明电影评论用词非常丰富
- **`<UNK>` 标记的作用：** 处理预测时遇到的未登录词（OOV），提高模型的泛化能力

### 训练过程的观察

从 `results.txt` 中可以看到：

```
Epoch:1, Loss:232.51, Val Acc:0.6440
Epoch:2, Loss:185.53, Val Acc:0.7496
Epoch:3, Loss:135.26, Val Acc:0.7832
Epoch:4, Loss:117.59, Val Acc:0.7840
Epoch:5, Loss:59.87, Val Acc:0.7912
```

**分析：**
- 第 1-2 轮损失下降明显，模型快速学习基本模式
- 第 3 轮后准确率增长放缓，可能接近当前架构的性能瓶颈
- 最终验证准确率 79.12%，对于这样一个简单模型来说表现不错
- 如果继续训练可能会过拟合，5 个 epoch 是合理的停止点

## 🚀 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

### 数据准备

下载 IMDB 数据集并解压到项目根目录：
```
aclImdb/
  └── train/
      ├── pos/
      └── neg/
```

### 训练模型

```bash
python train_imdb.py
```

训练完成后会生成 `imdb_lstm.pth` 模型文件。

### 预测

```bash
python predict.py
```

默认预测文本："This movie was surprisingly good and touching."  
你可以修改 `predict.py` 中的 `text` 变量来测试其他评论。

## 性能表现

- **训练集大小：** 25,000 条评论
- **验证集比例：** 10%
- **最终验证准确率：** 79.12%
- **词表规模：** 251,639 个唯一词汇
- **训练时间：** 约 10 分钟（取决于硬件）

## 🔧 模型架构

```
Input (Tokenized Text)
    ↓
Embedding Layer (vocab_size → 128)
    ↓
Bidirectional LSTM (128 → 128×2)
    ↓
Fully Connected Layer (256 → 2)
    ↓
Softmax (pos/neg)
```

**参数量统计：**
- Embedding: 251,639 × 128 ≈ 32M 参数
- LSTM: 约 0.3M 参数
- FC: 256 × 2 = 512 参数


## 代码结构

```
.
├── train_imdb.py       # 训练脚本（数据加载、模型定义、训练循环）
├── predict.py          # 预测脚本
├── requirements.txt    # 依赖库
├── results.txt         # 训练日志
├── imdb_lstm.pth       # 训练好的模型（训练后生成）
└── README.md           # 本文档
```

## 学习收获

通过这个项目，我深入理解了：

- 文本数据的预处理流程（分词、词表构建、序列填充）
- LSTM 在序列建模中的作用和局限性
- 训练循环的标准实现（前向传播、损失计算、反向传播）
- 如何评估模型性能并分析训练曲线
- PyTorch 的基本使用和模型保存/加载机制

## 参考资料

- [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 项目完成时间
- 2025.10.20
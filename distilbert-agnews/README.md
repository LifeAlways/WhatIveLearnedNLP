# AG News 文本分类项目

## 项目简介

这是一个基于 DistilBERT 的新闻文本分类项目，使用 AG News 数据集对新闻进行四分类（World、Sports、Business、Sci/Tech）。

## 核心思考与设计

### 1. 为什么选择 DistilBERT？

在模型选择上，权衡了性能与效率：

- **知识蒸馏的优势**：DistilBERT 保留了 BERT 97% 的语言理解能力，但参数量减少 40%，推理速度提升 60%，对于初学者和个人电脑更友好
- **资源友好**：相比完整的 BERT-base，DistilBERT 更适合在有限算力下快速迭代
- **任务适配性**：对于新闻分类这类相对清晰的任务，DistilBERT 的能力已经足够

### 2. 工程化的项目结构

我将项目拆分为独立的模块，每个脚本职责单一：

```
├── download_data.py      # 数据获取层
├── download_model.py     # 模型准备层
├── train_distilbert.py   # 训练核心逻辑
├── evaluate.py           # 评估与分析
├── predict.py            # 推理接口
└── requirements.txt      # 依赖管理
```

**设计理念**：
- 解耦数据/模型/训练流程，便于单独调试
- 网络环境隔离：支持镜像源切换和代理配置，便于在不同环境下下载
- 可复现性：本地化存储确保环境变化不影响实验

### 3. 数据处理

```python
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
```

- **固定长度策略**：设置 `max_length=256`，在训练效率与信息保留间取得平衡
- **批处理优化**：使用 `batched=True` 加速 tokenization
- **格式统一**：训练和评估使用完全一致的数据加载方式（Parquet 格式），避免数据不一致问题

### 4. 训练策略的考量

```python
TrainingArguments(
    eval_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    load_best_model_at_end=True,
)
```

**我的思考**：
- **轻量微调**：2 个 epoch 避免过拟合，AG News 数据集规模适中
- **学习率选择**：5e-5 是 BERT 系列模型的经典初始学习率
- **早停机制**：通过 `load_best_model_at_end` 自动选择验证集表现最佳的模型

## 实验结果

### 整体表现

| 指标 | 数值 |
|------|------|
| 测试集准确率 | **93.97%** |
| 宏平均 F1 | 0.94 |

### 分类别分析

| 类别 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| World | 0.96 | 0.94 | 0.95 | 1900 |
| Sports | 0.98 | 0.99 | 0.99 | 1900 |
| Business | 0.92 | 0.90 | 0.91 | 1900 |
| Sci/Tech | 0.90 | 0.93 | 0.91 | 1900 |

**结果洞察**：
- **Sports 类表现最优**（F1=0.99）：体育新闻用词特征明显（比赛、球员、得分等）
- **Business 和 Sci/Tech 略低**：两类可能存在主题交叉（如科技公司商业新闻）
- **类别均衡**：四个类别的 F1 差距不超过 0.08，说明模型没有明显偏向

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 完整流程

```bash
# 1. 下载数据（支持镜像源）
python download_data.py

# 2. 下载预训练模型（支持代理）
python download_model.py

# 3. 训练模型
python train_distilbert.py

# 4. 评估性能
python evaluate.py

# 5. 单条预测
python predict.py "Apple announces new iPhone with advanced AI features"
# 输出: 预测分类: Sci/Tech
```

## 项目特点

### 1. 网络环境适配
- 数据下载支持 Hugging Face 镜像源（国内可用）
- 模型下载支持 HTTP 代理配置
- 本地化存储策略，离线可复现

### 2. 完整的机器学习工作流
```
数据准备 → 模型下载 → 训练 → 评估 → 预测
```
每个环节独立脚本，可单独调试和优化

### 3. 生产级代码规范
- 路径统一管理（`DATA_DIR`、`MODEL_DIR`）
- 错误处理和日志输出
- 设备自动检测（CUDA/CPU）


## 技术栈

- **深度学习框架**：PyTorch 2.0+
- **预训练模型**：Hugging Face Transformers
- **数据处理**：Datasets, Tokenizers
- **评估指标**：Scikit-learn

## 学习收获

通过这个项目，我深入理解了：

1. **迁移学习的威力**：从零训练 → 使用预训练模型，准确率从 ~70% 跃升至 94%
2. **工程与算法的平衡**：好的代码结构让实验更高效
3. **评估的重要性**：不只看准确率，分类别的 Precision/Recall 能揭示模型短板

## 参考资料

- [DistilBERT 论文](https://arxiv.org/abs/1910.01108)
- [AG News 数据集](https://huggingface.co/datasets/ag_news)
- [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers)

## 项目完成时间
- 2025.12.01


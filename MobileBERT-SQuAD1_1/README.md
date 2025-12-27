# MobileBERT on SQuAD 1.1: 问答系统实现

> 一个基于MobileBERT的轻量级问答系统，在SQuAD 1.1数据集上进行微调

## 项目概述

本项目实现了一个完整的问答系统训练和推理流程，使用Google的MobileBERT模型在SQuAD 1.1数据集上进行微调。项目特别针对有限计算资源（如RTX 3050 4GB显存）进行了优化，并支持完全离线运行。

### 核心特性

- ✅ **轻量级模型**: 使用MobileBERT，参数量仅25M，适合资源受限环境
- ✅ **资源优化**: 针对3050显卡，4GB显存优化，通过梯度累积实现有效训练
- ✅ **完全离线**: 因国内网络受限，将加载与下载完全分开
- ✅ **端到端流程**: 涵盖数据下载、模型训练、评估和推理的完整pipeline

## 技术实现与思考

### 1. 模型选择思考

**为什么选择MobileBERT？**

在学习NLP的过程中，我发现很多教程都使用BERT-base或BERT-large，但这些模型对硬件要求较高。考虑到：
- **参数效率**: MobileBERT约25M参数，是BERT-base的1/4
- **推理速度**: 更适合部署到资源受限的环境
- **性能权衡**: 在SQuAD 1.1上仍能达到~88-90% F1分数

这让我意识到，在实际应用中，**模型大小与性能的平衡**是一个重要考量因素。

### 2. 训练优化策略

#### 显存优化
```python
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
# 有效batch_size = 16
```

**思考过程**：
- 初始尝试batch_size=8时显存溢出
- 通过梯度累积技术，将物理batch分成4个小批次
- 实现了相同的优化效果，但显存占用降低75%


#### 混合精度训练的取舍
```python
fp16 = False  # 暂时关闭以避免NaN问题
```

**思考**：
- FP16可以加速训练并减少显存占用
- 但在实际训练时出现了loss=0的情况，所以选择了关闭


### 3. 数据预处理的细节处理

在实现`prepare_train_features`函数时，我遇到了几个关键挑战：

#### 挑战1: 长文本的滑动窗口
```python
doc_stride = 128  # 滑动窗口步长
max_length = 384  # 最大序列长度
```

**问题**: SQuAD中的context可能超过384个token
**解决**: 使用滑动窗口切分，确保答案不会被截断

#### 挑战2: 答案位置映射
```python
# 找到答案在token序列中的位置
while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
    token_start_index += 1
answer_token_start = token_start_index - 1
```

**思考**：
- 字符级位置到token级位置的映射并非简单的索引转换
- 需要使用`offset_mapping`精确定位
- 对于跨window的答案，标记为`cls_token`（不可答）


### 4. 评估指标的理解

#### EM (Exact Match) vs F1 Score
- **EM**: 预测答案与标准答案完全一致的比例
- **F1**: 考虑部分匹配，更宽容的指标

**观察**：
- 本项目: EM=73.34%, F1=83.00%
- 预期性能: EM=80-82%, F1=88-90%

**分析与改进思路**：
1. **训练轮数**: 当前3轮可能不足，可尝试5-6轮
2. **学习率调度**: 当前使用固定warmup，可考虑cosine decay
3. **数据增强**: 可引入back-translation等技术
4. **后处理优化**: 改进答案提取策略

### 5. 离线训练的实现

```python
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
local_files_only = True
```

**动机**：
- 学习环境网络不稳定无法访问Hugging Face Hub
- 需要在完全离线的情况下复现实验

**实现思路**：
1. 预先下载所有数据和模型到本地缓存
2. 强制使用本地文件，避免任何网络请求
3. 提供清晰的错误提示和解决方案


## 项目结构

```
.
├── download_data.py      # 数据和模型下载脚本
├── train.py             # 训练主脚本
├── evaluate.py          # 模型评估脚本
├── predict.py           # 推理和演示脚本
├── requirements.txt     # 依赖项
├── data/               # 数据集缓存目录
├── models/             # 模型缓存目录
└── output/             # 训练输出和checkpoints
    └── final_model/    # 最终模型
```

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 如需GPU加速，确保安装CUDA版本的PyTorch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 第一步：下载数据和模型

```bash
python download_data.py
```

**功能**：
- 下载SQuAD 1.1数据集（训练集87k样本，验证集10k样本）
- 下载MobileBERT预训练模型
- 支持代理配置（如需要）

### 第二步：训练模型

```bash
python train.py
```

**训练配置**：
- 训练轮数: 3 epochs
- 批次大小: 4 (梯度累积4步，有效batch=16)
- 学习率: 3e-5
- 预计训练时间: ~3-4小时 (RTX 3050)

### 第三步：评估性能

```bash
# 完整评估
python evaluate.py

# 快速测试（100样本）
python evaluate.py --max_samples 100

# 显示预测示例
python evaluate.py --show_examples
```

### 第四步：推理测试

```bash
# 演示模式
python predict.py demo

# 交互模式
python predict.py
```

## 实验结果

### 性能指标

| 指标 | 本项目实现 | 预期性能 |
|------|-----------|---------|
| **Exact Match** | 73.34% | 80-82% |
| **F1 Score** | 83.00% | 88-90% |
| **训练时间** | ~2.5h | - |
| **模型大小** | ~100MB | - |

### 预测示例

```
问题: How much area does the Amazon rainforest cover?
上下文: The Amazon rainforest covers 5,500,000 km2...
预测: 5,500,000 km2
置信度: 100.00%
```


## 学习心得

通过这个项目，我获得了以下收获：

1. **理论到实践的差距**: 论文中的数字很漂亮，但实际复现时会遇到各种工程问题
2. **资源约束下的创新**: 有限的GPU显存迫使我深入理解梯度累积、混合精度等优化技术
3. **评估的重要性**: 单一指标不足以全面评价模型，需要结合多个角度分析
4. **端到端思维**: 从数据准备到模型部署，每个环节都需要细致考虑

## 依赖项

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- Datasets 2.14+
- 见`requirements.txt`

## 参考资料

- [MobileBERT论文](https://arxiv.org/abs/2004.02984)
- [SQuAD 1.1数据集](https://rajpurkar.github.io/SQuAD-explorer/)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers)

## 项目完成时间
- 2025.12.20


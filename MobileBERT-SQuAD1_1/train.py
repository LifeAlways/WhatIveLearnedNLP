"""
MobileBERT在SQuAD 1.1上的训练脚本
针对RTX 3050 4GB显存优化
"""

import os
import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    MobileBertForQuestionAnswering,
    MobileBertTokenizerFast,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint


# 配置参数
class Config:
    # 模型配置 - 使用本地缓存
    model_name = "google/mobilebert-uncased"
    model_cache_dir = "./models"  # 你的模型缓存目录
    use_offline = True  # 离线模式，不连接网络

    # 数据配置
    data_cache_dir = "./data"
    max_length = 384  # 最大序列长度
    doc_stride = 128  # 文档滑动窗口

    # 训练配置
    output_dir = "./output"
    num_train_epochs = 3
    per_device_train_batch_size = 4  # 可根据显存调整
    per_device_eval_batch_size = 8
    gradient_accumulation_steps = 4  # 有效batch_size = 16

    learning_rate = 3e-5
    weight_decay = 0.01
    warmup_steps = 500

    # 优化配置
    fp16 = False  # 暂时关闭混合精度，避免NaN问题
    # 如果训练正常，可以尝试开启 fp16=True 来加速
    gradient_checkpointing = False  # MobileBERT不支持梯度检查点

    # 梯度裁剪（防止梯度爆炸）
    max_grad_norm = 1.0

    # 日志和保存
    logging_steps = 100
    save_steps = 2000
    eval_steps = 2000
    save_total_limit = 2

    # 其他
    seed = 42
    resume_from_checkpoint = True


def prepare_train_features(examples, tokenizer, max_length, doc_stride):
    """
    预处理训练数据 - 修复版本
    """
    # 分词
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # 处理溢出和映射
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    # 初始化答案位置
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # 获取序列ID
        sequence_ids = tokenized.sequence_ids(i)

        # 获取对应的样本
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # 如果没有答案，标记为CLS
        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
            continue

        # 获取答案的字符位置
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # 找到context的token范围
        token_start_index = 0
        token_end_index = len(input_ids) - 1

        # 找到context部分的起始
        for idx in range(len(sequence_ids)):
            if sequence_ids[idx] == 1:
                token_start_index = idx
                break

        # 找到context部分的结束
        for idx in range(len(sequence_ids) - 1, -1, -1):
            if sequence_ids[idx] == 1:
                token_end_index = idx
                break

        # 如果答案不在当前window中，标记为CLS
        if not (offsets[token_start_index][0] <= start_char and
                offsets[token_end_index][1] >= end_char):
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
            continue

        # 找到答案的token起始位置
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        answer_token_start = token_start_index - 1

        # 找到答案的token结束位置
        while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        answer_token_end = token_end_index + 1

        # 最后验证：确保位置合理
        if (answer_token_start < 0 or
                answer_token_end >= len(input_ids) or
                answer_token_start > answer_token_end or
                answer_token_end - answer_token_start > 30):  # 答案不应该超过30个token
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            tokenized["start_positions"].append(answer_token_start)
            tokenized["end_positions"].append(answer_token_end)

    return tokenized


def main():
    """主训练函数"""
    print("=" * 60)
    print("MobileBERT + SQuAD 1.1 训练程序")
    print("=" * 60)

    config = Config()

    # 显示配置
    print(f"\n配置信息:")
    print(f"  模型: {config.model_name}")
    print(f"  训练轮数: {config.num_train_epochs}")
    print(f"  批次大小: {config.per_device_train_batch_size}")
    print(f"  梯度累积: {config.gradient_accumulation_steps}")
    print(f"  有效批次: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  混合精度: {config.fp16}")
    print(f"  最大长度: {config.max_length}")

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"\n✓ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("\n✗ 未检测到GPU，将使用CPU训练（会很慢）")

    # 加载数据集
    print("\n加载数据集...")
    try:
        # 设置离线模式（使用本地缓存）
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        print("  ✓ 数据集离线模式已启用")

        dataset = load_dataset("squad", cache_dir=config.data_cache_dir)
        print(f"✓ 数据集加载成功")
        print(f"  训练集: {len(dataset['train'])} 样本")
        print(f"  验证集: {len(dataset['validation'])} 样本")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        print("  请先运行 python download_data.py 下载数据")
        return

    # 加载模型和分词器
    print("\n加载模型...")
    try:
        # 设置离线模式（使用本地缓存）
        os.environ['TRANSFORMERS_OFFLINE'] = '1' if config.use_offline else '0'
        os.environ['HF_DATASETS_OFFLINE'] = '1' if config.use_offline else '0'

        if config.use_offline:
            print("  ✓ 离线模式已启用（使用本地缓存）")

        tokenizer = MobileBertTokenizerFast.from_pretrained(
            config.model_name,
            cache_dir=config.model_cache_dir,
            local_files_only=config.use_offline  # 只使用本地文件
        )
        model = MobileBertForQuestionAnswering.from_pretrained(
            config.model_name,
            cache_dir=config.model_cache_dir,
            local_files_only=config.use_offline  # 只使用本地文件
        )
        print(f"✓ 模型加载成功")

        # 注意：MobileBERT不支持梯度检查点
        # 如果显存不足，请减小batch_size而不是启用gradient_checkpointing

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("  请先运行 python download_data.py 下载模型")
        return

    # 预处理数据
    print("\n预处理数据...")
    tokenized_datasets = dataset.map(
        lambda x: prepare_train_features(
            x, tokenizer, config.max_length, config.doc_stride
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="预处理训练集"
    )
    print(f"✓ 数据预处理完成")

    # 训练配置（使用新版本参数）
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="steps",  # 新版本参数
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=config.seed,
        report_to="none",
        dataloader_pin_memory=False,
    )

    # 创建Trainer（使用新版本参数）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,  # 新版本参数
        data_collator=default_data_collator,
    )

    # 检查是否从checkpoint恢复
    last_checkpoint = None
    if config.resume_from_checkpoint and os.path.isdir(config.output_dir):
        last_checkpoint = get_last_checkpoint(config.output_dir)
        if last_checkpoint:
            print(f"\n✓ 找到checkpoint: {last_checkpoint}")
            print("  将从此处继续训练")

    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        # 保存模型
        print("\n保存模型...")
        trainer.save_model(os.path.join(config.output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(config.output_dir, "final_model"))

        # 保存训练指标
        metrics = train_result.metrics
        # 注意：以下方法在某些版本可能不可用，但不影响训练
        try:
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
        except AttributeError:
            # 如果方法不存在，手动保存metrics
            import json
            metrics_path = os.path.join(config.output_dir, "train_results.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"  训练指标已保存到: {metrics_path}")

        print("\n" + "=" * 60)
        print("✓ 训练完成!")
        print("=" * 60)
        print(f"\n训练指标:")
        print(f"  总步数: {metrics.get('train_steps', 'N/A')}")
        print(f"  训练损失: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"  训练时间: {metrics.get('train_runtime', 0) / 60:.1f} 分钟")
        print(f"\n模型保存位置: {os.path.join(config.output_dir, 'final_model')}")
        print("\n下一步:")
        print("  1. 运行 python predict.py 进行预测测试")
        print("  2. 运行 python evaluate.py 评估模型性能")

    except KeyboardInterrupt:
        print("\n\n训练被中断")
        print("可以稍后运行相同命令继续训练")
    except Exception as e:
        print(f"\n✗ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
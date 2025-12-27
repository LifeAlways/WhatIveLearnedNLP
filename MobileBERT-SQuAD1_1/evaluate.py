"""
评估MobileBERT在SQuAD 1.1验证集上的性能
计算EM (Exact Match) 和 F1 分数
完全离线模式 - 只使用本地缓存
"""

import os
import json
import string
import re
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import MobileBertForQuestionAnswering, MobileBertTokenizerFast

def normalize_answer(s):
    """
    规范化答案文本
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, ground_truth):
    """
    计算精确匹配分数
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    """
    计算F1分数
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    # 如果都为空
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0

    # 如果其中一个为空
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def get_answer(start_logits, end_logits, input_ids, tokenizer, max_answer_length=30):
    """
    从logits中提取答案
    """
    # 获取最可能的起始和结束位置
    start_indexes = torch.argsort(start_logits, descending=True)[:20]
    end_indexes = torch.argsort(end_logits, descending=True)[:20]

    valid_answers = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            if start_index <= end_index and end_index - start_index < max_answer_length:
                score = start_logits[start_index] + end_logits[end_index]
                valid_answers.append({
                    "start": start_index.item(),
                    "end": end_index.item(),
                    "score": score.item()
                })

    if not valid_answers:
        return ""

    # 选择得分最高的答案
    best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]

    # 提取答案文本
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_tokens = tokens[best_answer["start"]:best_answer["end"]+1]
    answer_text = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer_text

def evaluate_model(model_path, data_cache_dir="./data", max_samples=None):
    """
    评估模型

    Args:
        model_path: 模型路径
        data_cache_dir: 数据集缓存目录
        max_samples: 最大评估样本数（None表示全部）
    """
    print("="*60)
    print("SQuAD 1.1 模型评估")
    print("="*60)
    print()

    # 加载模型
    print("加载模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        tokenizer = MobileBertTokenizerFast.from_pretrained(
            model_path,
            local_files_only=True
        )
        model = MobileBertForQuestionAnswering.from_pretrained(
            model_path,
            local_files_only=True
        )
        model.to(device)
        model.eval()
        print("✓ 模型加载成功\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("  请确保模型路径正确")
        return

    # 加载数据集（完全离线）
    print("加载验证集...")
    try:
        # 设置离线环境变量
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        # 强制离线模式
        import datasets
        datasets.config.HF_DATASETS_OFFLINE = True

        print("  ✓ 离线模式已启用")

        # 加载数据集
        dataset = load_dataset(
            "squad",
            cache_dir=data_cache_dir,
            split="validation",
            download_mode="reuse_cache_if_exists"
        )

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"✓ 验证集加载成功: {len(dataset)} 样本\n")

    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        print("\n尝试查找本地缓存...")

        # 检查缓存目录
        cache_path = Path(data_cache_dir)
        if cache_path.exists():
            squad_dirs = list(cache_path.rglob("*squad*"))
            if squad_dirs:
                print(f"找到 {len(squad_dirs)} 个相关缓存文件/目录")
                print("但无法加载，请确保数据集已完整下载")
            else:
                print("未找到SQuAD缓存")
        else:
            print(f"缓存目录不存在: {data_cache_dir}")

        print("\n请先运行: python download_data.py")
        return

    # 评估
    print("开始评估...")
    print("-" * 60)

    exact_match_scores = []
    f1_scores = []

    for example in tqdm(dataset, desc="评估进度"):
        question = example["question"]
        context = example["context"]
        ground_truths = example["answers"]["text"]

        # 编码输入
        inputs = tokenizer(
            question,
            context,
            max_length=384,
            truncation="only_second",
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 预测
        with torch.no_grad():
            outputs = model(**inputs)

        # 提取答案
        prediction = get_answer(
            outputs.start_logits[0],
            outputs.end_logits[0],
            inputs["input_ids"][0],
            tokenizer
        )

        # 计算指标（对所有ground truth取最大值）
        em_scores = [compute_exact_match(prediction, gt) for gt in ground_truths]
        f1_score_list = [compute_f1(prediction, gt) for gt in ground_truths]

        exact_match_scores.append(max(em_scores))
        f1_scores.append(max(f1_score_list))

    # 计算平均分数
    avg_em = sum(exact_match_scores) / len(exact_match_scores) * 100
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100

    # 显示结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"\n样本数: {len(dataset)}")
    print(f"Exact Match (EM): {avg_em:.2f}%")
    print(f"F1 Score: {avg_f1:.2f}%")
    print()

    # 保存结果
    results = {
        "exact_match": avg_em,
        "f1": avg_f1,
        "num_samples": len(dataset)
    }

    results_path = os.path.join(model_path, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"结果已保存到: {results_path}")
    print()

    # 性能参考
    print("="*60)
    print("性能参考")
    print("="*60)
    print("MobileBERT在SQuAD 1.1上的预期性能:")
    print("  - EM: ~80-82%")
    print("  - F1: ~88-90%")
    print()

    if avg_f1 >= 88:
        print("✓ 模型性能达到预期!")
    elif avg_f1 >= 85:
        print("○ 模型性能接近预期，可以考虑:")
        print("  - 训练更多轮次")
        print("  - 调整学习率")
        print("  - 增加warmup步数")
    else:
        print("✗ 模型性能低于预期，建议:")
        print("  - 检查训练是否正常完成")
        print("  - 尝试重新训练")
        print("  - 调整超参数")

    return results

def show_predictions(model_path, data_cache_dir="./data", num_examples=5):
    """
    展示一些预测示例
    """
    print("\n" + "="*60)
    print("预测示例")
    print("="*60)

    # 加载模型（完全离线）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = MobileBertTokenizerFast.from_pretrained(
            model_path,
            local_files_only=True
        )
        model = MobileBertForQuestionAnswering.from_pretrained(
            model_path,
            local_files_only=True
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return

    # 加载数据（完全离线）
    try:
        # 设置离线环境变量
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        # 强制离线模式
        import datasets
        datasets.config.HF_DATASETS_OFFLINE = True

        dataset = load_dataset(
            "squad",
            cache_dir=data_cache_dir,
            split="validation",
            download_mode="reuse_cache_if_exists"
        )
    except Exception as e:
        print(f"✗ 无法加载数据集: {e}")
        print("跳过预测示例")
        return

    # 随机选择示例
    import random
    examples = random.sample(list(dataset), num_examples)

    for i, example in enumerate(examples, 1):
        print(f"\n示例 {i}:")
        print("-" * 60)
        print(f"上下文: {example['context'][:200]}...")
        print(f"问题: {example['question']}")
        print(f"真实答案: {example['answers']['text']}")

        # 预测
        inputs = tokenizer(
            example['question'],
            example['context'],
            max_length=384,
            truncation="only_second",
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = get_answer(
            outputs.start_logits[0],
            outputs.end_logits[0],
            inputs["input_ids"][0],
            tokenizer
        )

        print(f"预测答案: {prediction}")

        # 计算指标
        em = max([compute_exact_match(prediction, gt) for gt in example['answers']['text']])
        f1 = max([compute_f1(prediction, gt) for gt in example['answers']['text']])

        print(f"EM: {em}, F1: {f1:.2f}")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="评估MobileBERT模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/final_model",
        help="模型路径"
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default="./data",
        help="数据集缓存目录"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评估样本数（用于快速测试）"
    )
    parser.add_argument(
        "--show_examples",
        action="store_true",
        help="显示预测示例"
    )

    args = parser.parse_args()

    # 检查模型是否存在
    if not os.path.exists(args.model_path):
        print(f"✗ 模型不存在: {args.model_path}")
        print("  请先运行 python train.py 训练模型")
        return

    # 评估模型
    evaluate_model(
        args.model_path,
        args.data_cache_dir,
        args.max_samples
    )

    # 显示示例
    if args.show_examples:
        show_predictions(args.model_path, args.data_cache_dir)

if __name__ == "__main__":
    main()
import os
import torch
from transformers import MobileBertForQuestionAnswering, MobileBertTokenizerFast


class QAPredictor:
    def __init__(self, model_path="./output/final_model"):
        """
        初始化预测器

        Args:
            model_path: 训练好的模型路径
        """
        print("加载模型...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        try:
            self.tokenizer = MobileBertTokenizerFast.from_pretrained(model_path)
            self.model = MobileBertForQuestionAnswering.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("模型加载成功!\n")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("  请确保已经训练模型 (运行 python train.py)")
            raise

    def predict(self, question, context, top_k=1):
        """
        预测答案

        Args:
            question: 问题
            context: 上下文
            top_k: 返回前k个答案

        Returns:
            答案列表，每个答案包含文本和置信度
        """
        # 编码输入
        inputs = self.tokenizer(
            question,
            context,
            max_length=384,
            truncation="only_second",
            return_tensors="pt",
            padding=True
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取起始和结束位置的logits
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # 获取top-k个起始和结束位置
        start_indexes = torch.argsort(start_logits, descending=True)[:20]
        end_indexes = torch.argsort(end_logits, descending=True)[:20]

        # 生成候选答案
        valid_answers = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                # 确保答案合法
                if start_index <= end_index and end_index - start_index < 30:
                    score = start_logits[start_index] + end_logits[end_index]
                    valid_answers.append({
                        "start": start_index.item(),
                        "end": end_index.item(),
                        "score": score.item()
                    })

        # 按分数排序
        valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)

        # 提取答案文本
        results = []
        for answer in valid_answers[:top_k]:
            # 获取token到字符的映射
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            answer_tokens = tokens[answer["start"]:answer["end"] + 1]

            # 转换为文本
            answer_text = self.tokenizer.convert_tokens_to_string(answer_tokens)

            # 计算置信度（简单的softmax近似）
            confidence = torch.sigmoid(torch.tensor(answer["score"])).item()

            results.append({
                "text": answer_text,
                "confidence": confidence,
                "start": answer["start"],
                "end": answer["end"]
            })

        return results


def main():
    """主函数 - 交互式问答"""
    print("=" * 60)
    print("MobileBERT 问答系统")
    print("=" * 60)
    print()

    # 检查模型是否存在
    model_path = "./output/final_model"
    if not os.path.exists(model_path):
        print(f"模型不存在: {model_path}")
        print("  请先运行 python train.py 训练模型")
        return

    # 初始化预测器
    try:
        predictor = QAPredictor(model_path)
    except:
        return

    # 示例上下文
    default_context = """
    The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest 
    in the Amazon biome that covers most of the Amazon basin of South America. This basin 
    encompasses 7,000,000 km2 (2,700,000 sq mi), of which 5,500,000 km2 (2,100,000 sq mi) 
    are covered by the rainforest. The majority of the forest is contained within Brazil, 
    with 60% of the rainforest, followed by Peru with 13%, and Colombia with 10%.
    """

    print("示例上下文:")
    print("-" * 60)
    print(default_context.strip())
    print("-" * 60)
    print()

    # 交互式问答
    print("输入 'quit' 退出, 'context' 更换上下文")
    print()

    current_context = default_context

    while True:
        try:
            # 获取用户输入
            user_input = input("\n问题: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("再见!")
                break

            if user_input.lower() == 'context':
                print("\n请输入新的上下文 (输入空行结束):")
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                current_context = "\n".join(lines)
                print("\n上下文已更新")
                continue

            # 预测答案
            print("\n预测中...")
            answers = predictor.predict(user_input, current_context, top_k=3)

            # 显示结果
            print("\n" + "=" * 60)
            print("答案:")
            print("=" * 60)
            for i, answer in enumerate(answers, 1):
                print(f"\n{i}. {answer['text']}")
                print(f"   置信度: {answer['confidence']:.2%}")

        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"\n预测出错: {e}")


def demo():
    """演示函数 - 使用预定义的问题"""
    print("=" * 60)
    print("MobileBERT 问答演示")
    print("=" * 60)
    print()

    # 初始化预测器
    predictor = QAPredictor()

    # 测试样例
    test_cases = [
        {
            "context": "The Amazon rainforest covers 5,500,000 km2 and is mostly in Brazil with 60% of the forest.",
            "question": "How much area does the Amazon rainforest cover?"
        },
        {
            "context": "The Amazon rainforest covers 5,500,000 km2 and is mostly in Brazil with 60% of the forest.",
            "question": "Which country has most of the Amazon rainforest?"
        },
        {
            "context": "Super Bowl 50 was played on February 7, 2016, at Levi's Stadium in Santa Clara, California.",
            "question": "When was Super Bowl 50 played?"
        },
        {
            "context": "Super Bowl 50 was played on February 7, 2016, at Levi's Stadium in Santa Clara, California.",
            "question": "Where was Super Bowl 50 held?"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print("-" * 60)
        print(f"上下文: {test['context']}")
        print(f"问题: {test['question']}")

        answers = predictor.predict(test['question'], test['context'], top_k=1)

        print(f"\n答案: {answers[0]['text']}")
        print(f"置信度: {answers[0]['confidence']:.2%}")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
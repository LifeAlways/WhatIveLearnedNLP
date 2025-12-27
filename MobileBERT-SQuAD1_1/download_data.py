import os
import sys
from datasets import load_dataset
from transformers import MobileBertTokenizerFast, MobileBertForQuestionAnswering


# 设置代理
def setup_proxy(http_proxy=None, https_proxy=None):

    if http_proxy:
        os.environ['HTTP_PROXY'] = http_proxy
        os.environ['http_proxy'] = http_proxy
    if https_proxy:
        os.environ['HTTPS_PROXY'] = https_proxy
        os.environ['https_proxy'] = https_proxy

    print(f"代理设置:")
    print(f"  HTTP_PROXY: {os.environ.get('HTTP_PROXY', '未设置')}")
    print(f"  HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', '未设置')}")


def download_squad_dataset(save_dir="./data"):
    print("\n" + "=" * 50)
    print("开始下载SQuAD 1.1数据集...")
    print("=" * 50)

    try:
        # 下载数据集
        dataset = load_dataset("squad", cache_dir=save_dir)

        print(f"\n数据集下载成功!")
        print(f"  训练集大小: {len(dataset['train'])} 样本")
        print(f"  验证集大小: {len(dataset['validation'])} 样本")
        print(f"  保存位置: {save_dir}")

        # 显示示例
        print("\n数据集示例:")
        example = dataset['train'][0]
        print(f"  问题: {example['question']}")
        print(f"  上下文: {example['context'][:100]}...")
        print(f"  答案: {example['answers']}")

        return dataset

    except Exception as e:
        print(f"\n下载失败: {e}")
        sys.exit(1)


def download_mobilebert_model(model_name="google/mobilebert-uncased", save_dir="./models"):
    """下载MobileBERT模型和分词器"""
    print("\n" + "=" * 50)
    print(f"开始下载MobileBERT模型: {model_name}")
    print("=" * 50)

    try:
        # 下载分词器
        print("\n正在下载分词器...")
        tokenizer = MobileBertTokenizerFast.from_pretrained(
            model_name,
            cache_dir=save_dir
        )
        print("分词器下载成功!")

        # 下载模型
        print("\n正在下载模型...")
        model = MobileBertForQuestionAnswering.from_pretrained(
            model_name,
            cache_dir=save_dir
        )
        print("模型下载成功!")

        # 显示模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n模型信息:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
        print(f"  保存位置: {save_dir}")

        return tokenizer, model

    except Exception as e:
        print(f"\n下载失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    print("SQuAD 1.1 + MobileBERT 数据下载")


    setup_proxy(
         http_proxy="http://127.0.0.1:7890",
         https_proxy="http://127.0.0.1:7890")


    # 或者从环境变量读取
    proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    if proxy:
        print(f"\n检测到代理设置: {proxy}")
    else:
        print("\n未检测到代理设置")
        response = input("是否需要设置代理? (y/n): ")
        if response.lower() == 'y':
            proxy_url = input("请输入代理地址 (例如 http://127.0.0.1:7890): ")
            setup_proxy(proxy_url, proxy_url)

    # 下载数据集
    dataset = download_squad_dataset(save_dir="./data")

    # 下载模型
    tokenizer, model = download_mobilebert_model(
        model_name="google/mobilebert-uncased",
        save_dir="./models"
    )

    print("所有下载完成!")


if __name__ == "__main__":
    main()
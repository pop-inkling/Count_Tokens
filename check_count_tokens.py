from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

def main(dataset_name, dataset_config, dataset_split):
# 加载数据集，例如 wikitext 的 test 集
    if dataset_config is not None:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)

    # 加载 tokenizer，例如 LLaMA、GPT2、BERT 等
    tokenizer = AutoTokenizer.from_pretrained("/mnt/model/Llama-3.2-1B-Instruct")


    # 对整个数据集进行分词并统计 token 数量
    def count_tokens(example):
        return {"num_tokens": len(tokenizer(example["text"]).input_ids)}


    # 使用 map 添加 token 数
    dataset_with_token_counts = dataset.map(count_tokens)

    # 总 token 数
    total_tokens = sum(dataset_with_token_counts["num_tokens"])
    
    # 格式化显示token数量
    def format_tokens(num_tokens):
        if num_tokens >= 1e12:  # 1T
            return f"{num_tokens / 1e12:.2f}T"
        elif num_tokens >= 1e9:  # 1B
            return f"{num_tokens / 1e9:.2f}B"
        elif num_tokens >= 1e6:  # 1M
            return f"{num_tokens / 1e6:.2f}M"
        elif num_tokens >= 1e3:  # 1K
            return f"{num_tokens / 1e3:.2f}K"
        else:
            return str(num_tokens)
    
    formatted_tokens = format_tokens(total_tokens)
    print(f"Dataset: {dataset_name} :{dataset_config} : {dataset_split}")
    print(f"Total tokens: {total_tokens:,} ({formatted_tokens})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., wikitext")
    parser.add_argument("--config", type=str, default=None, help="Optional dataset config name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    args = parser.parse_args()

    main(args.dataset, args.config, args.split)

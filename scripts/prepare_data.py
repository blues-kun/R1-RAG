#!/usr/bin/env python3
"""
R1-RAG 数据准备脚本

此脚本:
1. 加载多跳问答数据集（HotpotQA, 2WikiMultihopQA等）
2. 使用GPT-4o生成黄金规划标注
3. 对照ground truth过滤标注
4. 以parquet格式导出训练数据

用法:
    python scripts/prepare_data.py --dataset hotpotqa --api_key YOUR_KEY
"""

import argparse
import os
import sys
from pathlib import Path

# 将项目根目录添加到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from r1_rag.data import PlanningDataProcessor, GPT4oPlanGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="准备R1-RAG训练数据")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "2wiki", "musique", "custom"],
        help="要处理的数据集"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="自定义JSONL输入文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/r1_rag",
        help="处理后数据的输出目录"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="用于GPT-4o标注的OpenAI API密钥（可选）"
    )
    parser.add_argument(
        "--generate_annotations",
        action="store_true",
        help="使用GPT-4o生成新标注"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="处理的最大样本数（用于测试）"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="训练数据的比例"
    )
    
    return parser.parse_args()


def load_dataset(dataset_name: str, input_file: str = None) -> PlanningDataProcessor:
    """根据名称或自定义文件加载数据集"""
    
    processor = PlanningDataProcessor(data_source=dataset_name)
    
    if input_file:
        # 从自定义文件加载
        processor.load_from_jsonl(input_file)
    else:
        # 从HuggingFace加载
        if dataset_name == "hotpotqa":
            processor.load_from_huggingface(
                "hotpot_qa",
                split="train",
                question_key="question",
                answer_key="answer"
            )
        elif dataset_name == "2wiki":
            processor.load_from_huggingface(
                "THUDM/2WikiMultihopQA",
                split="train"
            )
        elif dataset_name == "musique":
            processor.load_from_huggingface(
                "drt/musique",
                split="train"
            )
        else:
            raise ValueError(f"未知数据集: {dataset_name}")
    
    return processor


def main():
    args = parse_args()
    
    print("=" * 60)
    print("R1-RAG 数据准备")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 加载数据集
    print(f"\n[步骤1] 加载数据集: {args.dataset}")
    processor = load_dataset(args.dataset, args.input_file)
    
    # 打印统计信息
    stats = processor.get_statistics()
    print(f"\n数据集统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  有规划: {stats['with_plan']}")
    print(f"  有执行图: {stats['with_graph']}")
    
    # 如果请求则生成标注
    if args.generate_annotations:
        if not args.api_key:
            print("\n[错误] 标注生成需要API密钥")
            print("使用 --api_key YOUR_OPENAI_KEY 或设置 OPENAI_API_KEY 环境变量")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                sys.exit(1)
        else:
            api_key = args.api_key
        
        print(f"\n[步骤2] 生成GPT-4o标注...")
        processor.generate_annotations(
            api_key=api_key,
            max_samples=args.max_samples
        )
        
        # 更新统计信息
        stats = processor.get_statistics()
        print(f"\n标注后:")
        print(f"  有规划: {stats['with_plan']}")
        print(f"  有执行图: {stats['with_graph']}")
    
    # 过滤有标注的样本
    print(f"\n[步骤3] 过滤有标注的样本...")
    processor.filter_with_annotations()
    
    # 导出数据
    print(f"\n[步骤4] 导出到 {args.output_dir}")
    paths = processor.export_parquet(
        args.output_dir,
        train_ratio=args.train_ratio
    )
    
    # 同时导出JSONL用于检查
    processor.export_jsonl(os.path.join(args.output_dir, "data.jsonl"))
    
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print(f"  训练集: {paths['train']}")
    print(f"  验证集: {paths['val']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

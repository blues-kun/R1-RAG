"""
R1-RAG 规划数据处理器

处理数据流水线:
1. 加载原始多跳问答数据集
2. 生成或加载黄金规划标注
3. 构建正确格式的训练数据
4. 导出为parquet格式用于veRL训练
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm

from .prompts import PLANNING_PROMPT_TEMPLATE, ONE_SHOT_EXAMPLE
from .gpt4o_annotator import GPT4oPlanGenerator, AnnotationResult


@dataclass
class TrainingSample:
    """R1-RAG的单个训练样本"""
    id: int
    question: str
    golden_answers: List[str]
    data_source: str
    prompt: List[Dict[str, str]]
    ability: str = "multi-hop-reasoning"
    reward_model: Dict = None
    metadata: Dict = None
    extra_info: Dict = None
    
    def __post_init__(self):
        if self.reward_model is None:
            self.reward_model = {
                "ground_truth": {"target": self.golden_answers},
                "style": "rule"
            }
        if self.metadata is None:
            self.metadata = {}
        if self.extra_info is None:
            self.extra_info = {"index": self.id, "split": "train"}
    
    def to_dict(self) -> Dict:
        """转换为字典用于DataFrame"""
        return {
            "id": self.id,
            "question": self.question,
            "golden_answers": self.golden_answers,
            "data_source": self.data_source,
            "prompt": self.prompt,
            "ability": self.ability,
            "reward_model": self.reward_model,
            "metadata": self.metadata,
            "extra_info": self.extra_info
        }


class PlanningDataProcessor:
    """处理多跳问答数据用于R1-RAG训练
    
    支持两种模式:
    1. 使用GPT-4o生成新标注
    2. 从文件加载已有标注
    
    处理器确保:
    - 一致的prompt格式
    - 正确的元数据结构用于奖励计算
    - 基于答案验证的质量过滤
    """
    
    def __init__(
        self,
        data_source: str = "multi_hop_qa",
        use_one_shot: bool = True
    ):
        """初始化数据处理器
        
        Args:
            data_source: 数据源的名称标识
            use_one_shot: 是否在prompts中包含one-shot示例
        """
        self.data_source = data_source
        self.use_one_shot = use_one_shot
        self.samples: List[TrainingSample] = []
        
    def _format_prompt(self, question: str) -> List[Dict[str, str]]:
        """将问题格式化为聊天prompt格式
        
        Args:
            question: 原始问题文本
            
        Returns:
            包含格式化prompt的单个用户消息的列表
        """
        one_shot = ONE_SHOT_EXAMPLE if self.use_one_shot else ""
        
        prompt_text = PLANNING_PROMPT_TEMPLATE.format(
            one_shot_example=one_shot,
            question=question
        )
        
        return [{"role": "user", "content": prompt_text}]
    
    def load_from_jsonl(
        self,
        file_path: str,
        question_key: str = "question",
        answer_key: str = "golden_answers",
        plan_key: str = "plan",
        graph_key: str = "graph",
        hop_key: str = "hop"
    ) -> int:
        """从带有已有标注的JSONL文件加载数据
        
        Args:
            file_path: JSONL文件路径
            question_key: 问题字段的键
            answer_key: 黄金答案的键
            plan_key: 规划标注的键（可选）
            graph_key: 图标注的键（可选）
            hop_key: 跳数的键（可选）
            
        Returns:
            加载的样本数量
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f, desc="加载数据")):
                try:
                    data = json.loads(line.strip())
                    
                    question = data.get(question_key, "")
                    answers = data.get(answer_key, [])
                    if isinstance(answers, str):
                        answers = [answers]
                    
                    # 构建元数据
                    metadata = {}
                    if plan_key in data:
                        metadata["plan"] = data[plan_key]
                    if graph_key in data:
                        graph = data[graph_key]
                        if not isinstance(graph, list):
                            graph = [graph]
                        metadata["graph"] = graph
                    if hop_key in data:
                        metadata["hop"] = data[hop_key]
                    
                    sample = TrainingSample(
                        id=idx,
                        question=question,
                        golden_answers=answers,
                        data_source=self.data_source,
                        prompt=self._format_prompt(question),
                        metadata=metadata
                    )
                    self.samples.append(sample)
                    
                except Exception as e:
                    print(f"[警告] 解析第{idx}行失败: {e}")
                    continue
        
        print(f"[数据处理器] 从 {file_path} 加载了 {len(self.samples)} 个样本")
        return len(self.samples)
    
    def load_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        question_key: str = "question",
        answer_key: str = "golden_answers"
    ) -> int:
        """从HuggingFace数据集加载数据
        
        Args:
            dataset_name: HuggingFace数据集标识符
            split: 要加载的数据集分片
            question_key: 问题字段的键
            answer_key: 答案字段的键
            
        Returns:
            加载的样本数量
        """
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        
        for idx, item in enumerate(tqdm(dataset, desc="加载HF数据")):
            question = item.get(question_key, "")
            answers = item.get(answer_key, [])
            if isinstance(answers, str):
                answers = [answers]
            
            # 提取任何可用的元数据
            metadata = {}
            for key in ["plan", "graph", "hop", "metadata"]:
                if key in item:
                    metadata[key] = item[key]
            
            sample = TrainingSample(
                id=idx,
                question=question,
                golden_answers=answers,
                data_source=self.data_source,
                prompt=self._format_prompt(question),
                metadata=metadata
            )
            self.samples.append(sample)
        
        print(f"[数据处理器] 从 {dataset_name} 加载了 {len(self.samples)} 个样本")
        return len(self.samples)
    
    def generate_annotations(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_samples: Optional[int] = None
    ) -> int:
        """使用GPT-4o生成规划标注
        
        就地更新样本的标注。
        过滤掉标注生成失败的样本。
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的GPT模型
            max_samples: 标注的最大样本数（用于测试）
            
        Returns:
            成功标注的样本数量
        """
        annotator = GPT4oPlanGenerator(api_key=api_key, model=model)
        
        # 准备要标注的样本
        to_annotate = self.samples[:max_samples] if max_samples else self.samples
        
        raw_samples = [
            {"question": s.question, "golden_answers": s.golden_answers}
            for s in to_annotate
        ]
        
        # 生成标注
        results = annotator.generate_batch(raw_samples)
        
        # 用标注更新样本
        valid_samples = []
        for sample, result in zip(to_annotate, results):
            if result.is_valid:
                sample.metadata["plan"] = result.plan
                sample.metadata["graph"] = result.graph
                valid_samples.append(sample)
        
        # 替换为有效样本
        if max_samples:
            # 保留未标注的样本
            self.samples = valid_samples + self.samples[max_samples:]
        else:
            self.samples = valid_samples
        
        print(f"[数据处理器] 生成了 {len(valid_samples)} 个有效标注")
        return len(valid_samples)
    
    def filter_with_annotations(self) -> int:
        """过滤，只保留有规划标注的样本
        
        Returns:
            剩余样本数量
        """
        self.samples = [
            s for s in self.samples
            if s.metadata.get("plan") and s.metadata.get("graph")
        ]
        print(f"[数据处理器] 过滤后剩余 {len(self.samples)} 个有标注的样本")
        return len(self.samples)
    
    def export_parquet(
        self,
        output_path: str,
        train_ratio: float = 0.9
    ) -> Dict[str, str]:
        """导出处理后的数据为parquet文件
        
        Args:
            output_path: 保存parquet文件的目录
            train_ratio: 训练数据的比例（剩余用于验证）
            
        Returns:
            包含train和val文件路径的字典
        """
        os.makedirs(output_path, exist_ok=True)
        
        # 转换为DataFrame
        data = [s.to_dict() for s in self.samples]
        df = pd.DataFrame(data)
        
        # 分割train/val
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        # 保存
        train_path = os.path.join(output_path, "train.parquet")
        val_path = os.path.join(output_path, "val.parquet")
        
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        
        print(f"[数据处理器] 导出了 {len(train_df)} 个训练样本, {len(val_df)} 个验证样本")
        
        return {"train": train_path, "val": val_path}
    
    def export_jsonl(self, output_path: str) -> str:
        """导出处理后的数据为JSONL格式
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"[数据处理器] 导出了 {len(self.samples)} 个样本到 {output_path}")
        return output_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取加载数据的统计信息
        
        Returns:
            包含各种统计信息的字典
        """
        stats = {
            "total_samples": len(self.samples),
            "with_plan": sum(1 for s in self.samples if s.metadata.get("plan")),
            "with_graph": sum(1 for s in self.samples if s.metadata.get("graph")),
            "data_sources": list(set(s.data_source for s in self.samples))
        }
        
        # 跳数分布
        hop_counts = {}
        for s in self.samples:
            hop = s.metadata.get("hop", "unknown")
            hop_counts[hop] = hop_counts.get(hop, 0) + 1
        stats["hop_distribution"] = hop_counts
        
        return stats

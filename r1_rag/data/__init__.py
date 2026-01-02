"""
R1-RAG 数据模块

提供数据处理和标注生成:
- 多跳问答数据集加载（HotpotQA, 2Wiki, Musique）
- GPT-4o黄金规划标注生成
- 训练数据导出（parquet格式）
"""

from .processor import PlanningDataProcessor, TrainingSample
from .gpt4o_annotator import GPT4oPlanGenerator, AnnotationResult
from .prompts import (
    PLANNING_PROMPT_TEMPLATE,
    ONE_SHOT_EXAMPLE,
    GPT4O_PLAN_ANNOTATION_PROMPT,
)

__all__ = [
    # 数据处理
    "PlanningDataProcessor",
    "TrainingSample",
    
    # 标注生成
    "GPT4oPlanGenerator",
    "AnnotationResult",
    
    # Prompt模板
    "PLANNING_PROMPT_TEMPLATE",
    "ONE_SHOT_EXAMPLE",
    "GPT4O_PLAN_ANNOTATION_PROMPT",
]

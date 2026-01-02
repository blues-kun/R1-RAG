"""
R1-RAG: 推理优先的检索增强生成框架

一个用于多跳问答的强化学习框架，具有显式全局规划和密集过程监督。

核心特性:
- 基于DAG的规划结构，用于多跳推理
- 双重奖励机制：结构奖励(GED) + 语义奖励(E5相似度)
- 渐进式权重退火，实现平衡训练
- 兼容 Qwen2.5-3B-Instruct 骨干网络

核心模块:
- reward/: DAG奖励计算（图编辑距离 + E5语义）
- data/: 数据处理和GPT-4o标注
- agent/: 多轮搜索生成
- trainer/: GRPO训练与veRL集成
- retriever/: E5稠密检索服务

版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "R1-RAG Team"

# 奖励模块
from .reward import (
    DAGRewardEvaluator,
    RewardConfig,
    SemanticScorer,
    StructureScorer,
)

# 数据模块
from .data import (
    PlanningDataProcessor,
    GPT4oPlanGenerator,
    TrainingSample,
    AnnotationResult,
)

# Agent模块
from .agent import (
    LLMGenerationManager,
    GenerationConfig,
)

__all__ = [
    # 奖励相关
    "DAGRewardEvaluator",
    "RewardConfig",
    "SemanticScorer",
    "StructureScorer",
    
    # 数据相关
    "PlanningDataProcessor",
    "GPT4oPlanGenerator",
    "TrainingSample",
    "AnnotationResult",
    
    # Agent相关
    "LLMGenerationManager",
    "GenerationConfig",
]

"""
R1-RAG 奖励模块

提供多维度奖励计算用于强化学习训练:
- DAG结构相似度（图编辑距离）
- 语义相似度（E5嵌入）
- 子目标完成准确度（F1分数）
- 渐进式权重调度
"""

from .dag_evaluator import DAGRewardEvaluator
from .config import RewardConfig
from .semantic_scorer import SemanticScorer
from .structure_scorer import StructureScorer

__all__ = [
    "DAGRewardEvaluator",
    "RewardConfig",
    "SemanticScorer", 
    "StructureScorer",
]

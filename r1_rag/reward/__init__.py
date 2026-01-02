"""
Reward Module for R1-RAG

Provides multi-dimensional reward computation for RL training:
- DAG structural similarity (Graph Edit Distance)
- Semantic similarity (E5 embedding)  
- Sub-goal completion accuracy (F1 score)
- Progressive weight scheduling
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


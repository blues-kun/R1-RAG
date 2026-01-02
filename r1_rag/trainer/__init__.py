"""
R1-RAG 训练模块

提供基于DAG过程监督的GRPO训练。
"""

from .main_grpo import GRPOTrainer, RewardManager

__all__ = [
    "GRPOTrainer",
    "RewardManager",
]

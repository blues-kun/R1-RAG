"""
R1-RAG Trainer Module

Provides GRPO training with DAG-based process supervision.
"""

from .main_grpo import GRPOTrainer, RewardManager

__all__ = [
    "GRPOTrainer",
    "RewardManager",
]


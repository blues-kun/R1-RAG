"""
R1-RAG: Reasoning-First Retrieval-Augmented Generation

A reinforcement learning framework for multi-hop question answering
with explicit global planning and dense process supervision.

Key Features:
- DAG-based planning structure for multi-hop reasoning
- Dual reward mechanism: structural (GED) + semantic (E5 similarity)
- Progressive weight annealing for balanced training
- Compatible with Qwen2.5-3B-Instruct backbone

Core Components:
- reward/: DAG-based reward computation (GED + E5)
- data/: Data processing and GPT-4o annotation
- agent/: Multi-turn generation with search
- trainer/: GRPO training integration with veRL
- retriever/: E5-based dense retrieval server

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Reward Module
from .reward import (
    DAGRewardEvaluator,
    RewardConfig,
    SemanticScorer,
    StructureScorer,
)

# Data Module
from .data import (
    PlanningDataProcessor,
    GPT4oPlanGenerator,
    TrainingSample,
    AnnotationResult,
)

# Agent Module
from .agent import (
    LLMGenerationManager,
    GenerationConfig,
)

__all__ = [
    # Reward
    "DAGRewardEvaluator",
    "RewardConfig",
    "SemanticScorer",
    "StructureScorer",
    
    # Data
    "PlanningDataProcessor",
    "GPT4oPlanGenerator",
    "TrainingSample",
    "AnnotationResult",
    
    # Agent
    "LLMGenerationManager",
    "GenerationConfig",
]

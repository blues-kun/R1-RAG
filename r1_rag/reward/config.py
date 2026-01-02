"""
Reward Configuration for R1-RAG

Defines hyperparameters for the dual reward mechanism:
- Structural reward weights
- Semantic similarity thresholds
- Progressive annealing schedule
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardConfig:
    """Configuration for DAG-based reward computation.
    
    The reward combines three signals:
    1. Answer correctness (outcome-based)
    2. Planning quality (process-based, structural)
    3. Sub-goal completion (process-based, semantic)
    
    Progressive annealing gradually shifts focus from process to outcome.
    """
    
    # Embedding model for semantic similarity
    embedding_model_path: str = "intfloat/e5-base-v2"
    
    # Similarity thresholds
    node_match_threshold: float = 0.7  # Min cosine similarity to match nodes
    
    # Reward component weights
    format_weight: float = 0.1      # Weight for format compliance
    plan_sim_weight: float = 0.5    # Weight for plan semantic similarity
    structure_weight: float = 0.5   # Weight for DAG structure match (GED)
    step_weight: float = 0.5        # Weight for sub-goal completion
    
    # Graph Edit Distance parameters
    ged_beta: float = 1.0           # GED normalization factor
    plan_alpha: float = 0.9         # Balance between semantic and structural
    
    # Progressive annealing schedule
    annealing_total_steps: int = 50     # Total training steps for annealing
    annealing_center: float = 0.9       # Center point (90% of total steps)
    annealing_temperature: float = 10.0 # Controls transition steepness
    
    # Validation mode (only use answer score)
    validation_mode: bool = False
    
    def get_annealing_weight(self, current_step: int) -> float:
        """Calculate progressive annealing weight.
        
        Early training (weight ≈ 1): Focus on planning quality
        Late training (weight → 0): Focus on answer correctness
        
        Formula: w(t) = 1 / (1 + exp((t - center*T) / k))
        """
        import numpy as np
        t = float(current_step)
        T = float(self.annealing_total_steps)
        k = self.annealing_temperature
        center = self.annealing_center
        
        return 1.0 / (1.0 + np.exp((t - center * T) / k))


@dataclass  
class DAGNodeConfig:
    """Configuration for DAG node representation."""
    
    # Placeholder pattern for dependencies
    placeholder_pattern: str = r"<A(\d+)>"
    
    # Node ID prefix
    node_prefix: str = "Q"
    
    # Answer placeholder format
    answer_placeholder: str = "<A{index}>"


"""
R1-RAG Data Module

Provides data processing and annotation generation:
- Multi-hop QA dataset loading (HotpotQA, 2Wiki, Musique)
- GPT-4o golden plan annotation generation
- Training data export (parquet format)
"""

from .processor import PlanningDataProcessor, TrainingSample
from .gpt4o_annotator import GPT4oPlanGenerator, AnnotationResult
from .prompts import (
    PLANNING_PROMPT_TEMPLATE,
    ONE_SHOT_EXAMPLE,
    GPT4O_PLAN_ANNOTATION_PROMPT,
)

__all__ = [
    # Data Processing
    "PlanningDataProcessor",
    "TrainingSample",
    
    # Annotation Generation
    "GPT4oPlanGenerator",
    "AnnotationResult",
    
    # Prompt Templates
    "PLANNING_PROMPT_TEMPLATE",
    "ONE_SHOT_EXAMPLE",
    "GPT4O_PLAN_ANNOTATION_PROMPT",
]

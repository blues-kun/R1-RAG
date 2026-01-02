"""
R1-RAG Agent Module

Provides multi-turn generation management:
- LLM generation loop with search integration
- Action parsing and state management
- GPU-aware batch processing
"""

from .generation_manager import (
    LLMGenerationManager,
    GenerationConfig,
    TensorHelper,
)

__all__ = [
    "LLMGenerationManager",
    "GenerationConfig",
    "TensorHelper",
]

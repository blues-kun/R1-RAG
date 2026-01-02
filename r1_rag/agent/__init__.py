"""
R1-RAG Agent模块

提供多轮生成管理:
- 带搜索集成的LLM生成循环
- 动作解析和状态管理
- GPU感知的批处理
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

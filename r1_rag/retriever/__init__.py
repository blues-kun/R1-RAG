"""
R1-RAG 检索模块

提供搜索增强生成的检索服务:
- E5稠密检索服务器
- BM25稀疏检索（可选）
- 文档索引工具
"""

from .server import RetrievalServer, app
from .indexer import DocumentIndexer

__all__ = [
    "RetrievalServer",
    "DocumentIndexer",
    "app",
]

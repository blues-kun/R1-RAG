"""
R1-RAG Retriever Module

Provides retrieval services for search-augmented generation:
- E5-based dense retrieval server
- BM25 sparse retrieval (optional)
- Document indexing utilities
"""

from .server import RetrievalServer, app
from .indexer import DocumentIndexer

__all__ = [
    "RetrievalServer",
    "DocumentIndexer",
    "app",
]


"""
Retrieval Server for R1-RAG

FastAPI-based retrieval service that:
1. Loads pre-built FAISS index
2. Handles batch retrieval requests
3. Returns top-k relevant documents

Optimized for multi-turn RAG training.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# ============== Request/Response Models ==============

class RetrievalRequest(BaseModel):
    """Request model for retrieval endpoint."""
    queries: List[str]
    topk: int = 3
    return_scores: bool = True


class DocumentResult(BaseModel):
    """Single document result."""
    docid: str
    contents: str
    score: Optional[float] = None


class RetrievalResponse(BaseModel):
    """Response model for retrieval endpoint."""
    result: List[List[Dict[str, Any]]]
    query_count: int


# ============== Retrieval Server ==============

class RetrievalServer:
    """E5-based dense retrieval server.
    
    Features:
    - Efficient batch encoding with E5 model
    - FAISS index for fast similarity search
    - Document corpus loading from JSONL
    - GPU acceleration when available
    """
    
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        model_path: str = "intfloat/e5-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize retrieval server.
        
        Args:
            index_path: Path to FAISS index file
            corpus_path: Path to document corpus JSONL
            model_path: E5 model identifier or path
            device: Device for encoding (cuda/cpu)
        """
        self.device = device
        self.model_path = model_path
        
        # Load components
        print(f"[Retriever] Loading E5 model: {model_path}")
        self._load_encoder()
        
        print(f"[Retriever] Loading index: {index_path}")
        self._load_index(index_path)
        
        print(f"[Retriever] Loading corpus: {corpus_path}")
        self._load_corpus(corpus_path)
        
        print(f"[Retriever] Ready! Corpus size: {len(self.corpus)}")
    
    def _load_encoder(self):
        """Load E5 sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        
        self.encoder = SentenceTransformer(self.model_path, device=self.device)
    
    def _load_index(self, index_path: str):
        """Load FAISS index from file."""
        import faiss
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Move to GPU if available
        if self.device == "cuda":
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("[Retriever] Index moved to GPU")
            except Exception as e:
                print(f"[Retriever] GPU index failed, using CPU: {e}")
    
    def _load_corpus(self, corpus_path: str):
        """Load document corpus from JSONL file."""
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        
        self.corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                self.corpus.append(doc)
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries into dense vectors.
        
        Args:
            queries: List of query strings
            
        Returns:
            Query embeddings of shape (N, dim)
        """
        # E5 requires "query: " prefix for queries
        prefixed = [f"query: {q}" for q in queries]
        embeddings = self.encoder.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
    
    def search(
        self,
        queries: List[str],
        topk: int = 3,
        return_scores: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """Search for relevant documents.
        
        Args:
            queries: List of query strings
            topk: Number of results per query
            return_scores: Whether to include similarity scores
            
        Returns:
            List of result lists, one per query
        """
        # Encode queries
        query_vectors = self.encode_queries(queries)
        
        # Search index
        scores, indices = self.index.search(query_vectors, topk)
        
        # Format results
        results = []
        for query_idx, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            query_results = []
            for score, doc_idx in zip(query_scores, query_indices):
                if doc_idx < 0 or doc_idx >= len(self.corpus):
                    continue
                    
                doc = self.corpus[doc_idx]
                result = {
                    "docid": doc.get("id", str(doc_idx)),
                    "document": {
                        "contents": doc.get("contents", doc.get("text", "")),
                    }
                }
                if return_scores:
                    result["score"] = float(score)
                    
                query_results.append(result)
            
            results.append(query_results)
        
        return results


# ============== FastAPI Application ==============

app = FastAPI(
    title="R1-RAG Retrieval Server",
    description="E5-based dense retrieval for multi-hop QA",
    version="1.0.0",
)

# Global server instance
_server: Optional[RetrievalServer] = None


def get_server() -> RetrievalServer:
    """Get the global retrieval server instance."""
    if _server is None:
        raise HTTPException(
            status_code=503, 
            detail="Retrieval server not initialized"
        )
    return _server


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """Retrieve relevant documents for queries.
    
    Args:
        request: RetrievalRequest with queries and parameters
        
    Returns:
        RetrievalResponse with search results
    """
    server = get_server()
    
    try:
        results = server.search(
            queries=request.queries,
            topk=request.topk,
            return_scores=request.return_scores,
        )
        
        return RetrievalResponse(
            result=results,
            query_count=len(request.queries),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "corpus_size": len(_server.corpus) if _server else 0}


# ============== CLI Entry Point ==============

def main():
    """Run retrieval server from command line."""
    global _server
    
    parser = argparse.ArgumentParser(description="R1-RAG Retrieval Server")
    parser.add_argument("--index_path", type=str, required=True, help="Path to FAISS index")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus JSONL")
    parser.add_argument("--model_path", type=str, default="intfloat/e5-base-v2", help="E5 model path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    # Initialize server
    _server = RetrievalServer(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_path=args.model_path,
    )
    
    # Run FastAPI
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

